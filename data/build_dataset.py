from bs4 import BeautifulSoup
from os import listdir
import random
import math
import argparse
import spacy
import cssutils

nlp = spacy.load('fr')

dict_file = "token.dict"
TOKEN_DICT = {}
with open(dict_file, "r") as f:
    index = 0
    for line in f.readlines():
        TOKEN_DICT[line.strip()] = index
        index += 1

def get_html_files_from_folder(folder):
    """ Get all html files from folder """
    return [f for f in listdir(folder) if ".html" in f]

def get_gt_files_from_folder(folder):
    """ Get all gt files from folder """
    return [f for f in listdir(folder) if ".gt" in f]

def extract_text_from_html_element(element):
    """ Get text from html element """
    return element.text

def create_style_to_class_from_css(cssfile):
    style_to_class = {
        "underlined": [],
        "bold": [],
        "italic": []
    }
    sheet = cssutils.parseFile(cssfile)
    for rule in sheet:
        if rule.type != rule.STYLE_RULE:
            continue

        selector = rule.selectorText
        if not selector[0] == ".":
            continue

        selector = selector.split(" ")[0][1:]

        text_decoration_property = rule.style.getProperty("text-decoration")
        if text_decoration_property and text_decoration_property.value == "underline":
            style_to_class["underlined"].append(selector)

        font_weight_property = rule.style.getProperty("font-weight")
        if font_weight_property and font_weight_property.value == "bold":
            style_to_class["bold"].append(selector)

        font_style_property = rule.style.getProperty("font-style")
        if font_style_property and font_style_property.value == "italic":
            style_to_class["italic"].append(selector)

    return style_to_class

def extract_style_features_from_html_element(element, style_to_class):
    """Extract some style features based on css."""

    output = ["0" for index in range(len(style_to_class))]

    if "bold" in style_to_class and element.name == 'h2':
        output[list(style_to_class.keys()).index("bold")] = "1"

    if "underlined" in style_to_class and element.find("a"):
        output[list(style_to_class.keys()).index("underlined")] = "1"

    if not element.has_attr('class'):
        return output

    element_class = element["class"]
    for index, style in enumerate(style_to_class.keys()):
        for html_class in element_class:
            if html_class in style_to_class[style]:
                output[index] = "1"
                break

    return output

def extract_token_features_from_html_element(element):
    """Get text vectorization based on token dict"""
    text = extract_text_from_html_element(element)
    token = [t.lemma_ for t in nlp(text)]

    output = ["0" for index in range(len(TOKEN_DICT))]
    for tok in token:
        if tok.lower() in TOKEN_DICT.keys():
            output[TOKEN_DICT[tok.lower()]] = "1"
    return output

def extract_general_features_from_html_element(element):
    """ Get features from html element """
    text = extract_text_from_html_element(element)
    doc = nlp(text)

    # Num of words
    num_words = "{:.2f}".format(min(len(doc), 50)/50.)

    return [num_words]

def get_text_and_features(filename, style_to_class):
    """ Open html file and get all text and associated features """

    # build html tree from file
    with open(filename, encoding="cp1252") as f:
        soup = BeautifulSoup(f, "html.parser")

    document = soup.find(id="contentDocument")

    if not document:
        print(filename)

    file_text = []
    file_features = []
    for paragraph in document.find_all(["h1", "h2", "h3", "h4", "p"]):
        text = extract_text_from_html_element(paragraph)

        if not text.strip():
            continue

        features = extract_general_features_from_html_element(paragraph)

        if style_to_class:
            features += extract_style_features_from_html_element(paragraph, style_to_class)

        features += extract_token_features_from_html_element(paragraph)

        file_text.append(text)
        file_features.append(features)

    return file_text, file_features

def get_groundtruth(gtfilename):
    """ Open groundtruth file and get all gt """

    gt_array = []
    with open(gtfilename, "r") as f:
        for line in f.readlines():
            index, gt, _ = line.split("\t")
            gt_array.append(gt)

    return gt_array

def prepare_gt_files(raw_files, gt_folder, style_to_class):
    """ Create groundtruth files from list of files """

    gt_files = get_gt_files_from_folder(gt_folder)
    gt_files = ["{}/{}".format(gt_folder, filename) for filename in gt_files]

    for filename in raw_files:
        gt_name = filename.replace("html", "gt")

        if gt_name in gt_files:
            continue

        else:
            with open(gt_name, "w") as f_gt:
                text, features = get_text_and_features(filename, style_to_class)
                for index in range(len(text)):
                    f_gt.write("{}\t  0\t{}\n".format(str(index).rjust(3), text[index]))

def build_dataset(name, raw_files, style_to_class):
    """ Create dataset from list of files """

    with open("{}.text".format(name), "w") as f_text:
        with open("{}.feat".format(name), "w") as f_features:

            for filename in raw_files:
                text, features = get_text_and_features(filename, style_to_class)

                gt = get_groundtruth("gt/{}".format(filename.split("/")[-1].replace("html", "gt")))

                short_filename = filename.split("/")[-1].split(".")[0]

                for index in range(len(text)):
                    f_text.write("{}\t{}\t{}\n".format(short_filename, index, text[index]))
                f_text.write("\n")

                for index in range(len(features)):
                    f = "\t".join(features[index])
                    f_features.write("{}\t{}\t{}\t{}\n".format(short_filename, index, gt[index], f))
                f_features.write("\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt", default=False, help="create groundtruth")
    parser.add_argument("--bt", default=False, help="create bootstrap dataset")
    args = parser.parse_args()

    gt_folder = "./gt"
    raw_folder = "./html"
    bootstrap_folder = "./html/bootstrap"
    dev_dataset = 0.2
    test_dataset = 0.2

    if args.gt or args.bt:
        raw_files = get_html_files_from_folder(bootstrap_folder)
        raw_files = ["{}/{}".format(bootstrap_folder, filename) for filename in raw_files]
    else:
        raw_files = get_html_files_from_folder(raw_folder)
        raw_files = ["{}/{}".format(raw_folder, filename) for filename in raw_files]
        random.shuffle(raw_files)

    if args.gt:
        print("Create groundtruth...")
        prepare_gt_files(raw_files, gt_folder)

    elif args.bt:
        style_to_class = create_style_to_class_from_css("./html/style.css")
        print("Prepare useful data...")
        build_dataset("bootstrap", raw_files, style_to_class)

    else:
        print("Prepare useful data...")
        style_to_class = create_style_to_class_from_css("./html/style.css")

        print("Create dataset...")
        num_files = len(raw_files)
        num_dev_files = math.floor(dev_dataset * num_files)
        num_test_files = math.floor(test_dataset * num_files)

        dev_files = raw_files[:num_dev_files]
        test_files = raw_files[num_dev_files : num_dev_files + num_test_files]
        train_files = raw_files[num_dev_files + num_test_files:]

        build_dataset("train", train_files, style_to_class)
        build_dataset("test", test_files, style_to_class)
        build_dataset("dev", dev_files, style_to_class)
