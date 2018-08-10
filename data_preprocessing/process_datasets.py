from entailment.data_preprocessing.text_to_kg_generator import SpacyDictBasedEntityExtractor, EntityLinking
from entailment.knowledge_graph.fact_generator import FactGenerator, OneHopFactGenerator, TwoHopPathBasedFactGenerator
from entailment.data_preprocessing.text_to_kg_generator import Extractor, ConceptNetEntityExtractorLinker, \
    TextToSparqlGraphGenerator, TextToGraphGenerator, DBpediaSpotlightExtractor
import ntpath
import os
import argparse
import socket
import multiprocessing
from functools import partial
import concurrent.futures


class entailment_instance:
    def __init__(self, premise: str, hypothesis: str, label: str = None):
        self.premise = premise
        self.hypothesis = hypothesis
        self.premise_links = []
        self.hypothesis_links = []
        self.premise_entities = []
        self.hypothesis_entities = []
        self.premise_triples = []
        self.hypothesis_triples = []
        self.label = label

    def update_conceptnet_annotations(self, entities: list, premise: bool = True):
        if premise:
            self.premise_entities = entities
        else:
            self.hypothesis_entities = entities

    def update_conceptnet_triples(self, triples: list, premise: bool = True):
        if premise:
            self.premise_triples = triples
        else:
            self.hypothesis_triples = triples

    def update_conceptnet_links(self, links: list, premise: bool = True):
        if premise:
            self.premise_links = links
        else:
            self.hypothesis_links = links


def path_leaf(path):
    # Returns the filename from the path
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)


def process_scitail_line(line: str):
    # Process to get premise hypothesis and label here
    line = line.strip("\n")
    line_split = line.split('\t')
    if len(line_split) < 3:
        raise ValueError("Does not match the number of columns required")
    premise, hypothesis, label = line_split
    return premise, hypothesis, label


def process_snli_format_line(line: str):
    line = line.strip("\n")
    line_split = line.split('\t')
    if len(line_split) < 14:
        raise ValueError("Does not match the number of columns required")
    label, sentence1_binary_parse, sentence2_binary_parse, sentence1_parse, \
        sentence2_parse, premise, hypothesis, captionID, pairID,\
        label1, label2, label3, label4, label5 = line_split
    return premise, hypothesis, label


def process_multi_nli_format_line(line: str):
    line = line.strip("\n")
    line_split = line.split('\t')
    if len(line_split) < 14:
        raise ValueError("Does not match the number of columns required")
    label, sentence1_binary_parse, sentence2_binary_parse, sentence1_parse, \
        sentence2_parse, premise, hypothesis, promptID, pairID, genre, \
        label1, label2, label3, label4, label5 = line_split
    return premise, hypothesis, label


def isDataFile(filename: str):
    sub_str = ['test', 'train', 'dev']
    for s in sub_str:
        if s in path_leaf(filename).lower():
            return True
    return False

def process_entity_files(entity_file: str, entailment_file: str, output_dir: str,
                         file_format: "scitail", kg_sparql_endpoint=None):
    sentence_entities = {}
    sentence_links = {}
    linker = ConceptNetEntityExtractorLinker("../data/conceptnet")
    file_str = path_leaf(entailment_file).replace(".tsv",'').replace(".txt",'')
    with open(entity_file, "r") as entity_f:
        for line in entity_f.readlines():
            sentence, entity_csv = line.split("\t")
            entities = entity_csv.split(",")
            space_stripped_entities = []
            links = []
            for entity in entities:
                if entity.strip() != '':
                    space_stripped_entities.append(entity.strip())
                    links.extend(linker.link_entity(entity.strip()))
            sentence_entities[sentence] = space_stripped_entities
            sentence_links[sentence] = links
    existing_processed = []
    if os.path.isfile(output_dir+file_str+"_entities.tsv"):
        with open(output_dir+file_str+"_entities.tsv", 'r') as existing_processed_file:
            for line in existing_processed_file:
                p, p_e, h, h_e, l = line.split("\t")
                existing_processed.append(p+h)
    print("Number of pairs already processed: " + str(len(existing_processed)))
    entailment_instances = []
    with open(entailment_file, "r") as scitail_file:
        skipped = 0
        for line in scitail_file.readlines():
            if file_format == "snli":
                premise, hypothesis, label = process_snli_format_line(line)
            elif file_format == "scitail":
                premise, hypothesis, label = process_scitail_line(line)
            elif file_format == "multinli":
                premise, hypothesis, label = process_multi_nli_format_line(line)
            if premise+hypothesis in existing_processed:
                skipped += 1
                continue
            e = entailment_instance(premise=premise, hypothesis=hypothesis, label=label)
            try:
                premise_entities = sentence_entities[premise]
                premise_links = sentence_links[premise]
                hypothesis_entities = sentence_entities[hypothesis]
                hypothesis_links = sentence_links[hypothesis]
            except KeyError:
                print("One of them is missing entities.")
                print("Premise: " + premise)
                print("Hypothesis: " + hypothesis)
                continue
            e.update_conceptnet_annotations(premise_entities, premise=True)
            e.update_conceptnet_links(premise_links, premise=True)
            e.update_conceptnet_annotations(hypothesis_entities, premise=False)
            e.update_conceptnet_links(hypothesis_links, premise=False)
            entailment_instances.append(e)
    print("Number of skipped pairs: " + str(skipped))
    print("Number of pairs to be processed: " + str(len(entailment_instances)))
    factgenerator = TwoHopPathBasedFactGenerator(sparql_endpoint=kg_sparql_endpoint, destination_provided=True)
    NUM_WORKERS = 1
    func = partial(process_each_inst, factgenerator, output_dir+file_str)
    with multiprocessing.Pool(processes=NUM_WORKERS) as pool:
        results = pool.map_async(func, entailment_instances)
        results.wait()

def process_each_inst(factgen, output_dir, e_inst=None):

    entities_file = open(output_dir + "_entities.tsv", "a")
    triples_file = open(output_dir + "_triples.tsv", "a")
    no_entities_file = open(output_dir + "_no_entities.tsv", "a")
    triples_entities_file = open(output_dir + "_triple_entities.tsv", "a")

    entities = {}
    entities['source'] = e_inst.premise_links
    #dest = e_inst.premise_links
    dest = e_inst.hypothesis_links
    entities['destination'] = dest
    #print(e_inst.premise)
    e_inst.premise_triples = factgen.generate_facts(entities)

    entities = {}
    entities['source'] = e_inst.hypothesis_links
    #dest = e_inst.hypothesis_links
    dest = e_inst.premise_links
    entities['destination'] = dest
    #print(e_inst.hypothesis)
    e_inst.hypothesis_triples = factgen.generate_facts(entities)

    if e_inst.premise_entities is None or len(e_inst.premise_entities) == 0:
        no_entities_file.write("P " + e_inst.premise + "\t" + e_inst.hypothesis + e_inst.label + "\n")
        return
    if e_inst.hypothesis_entities is None or len(e_inst.hypothesis_entities) == 0:
        no_entities_file.write("H " + e_inst.premise + "\t" + e_inst.hypothesis + "\t" + e_inst.label + "\n")
        return

    entities_file.write(e_inst.premise + "\t")
    for i in range(0, len(e_inst.premise_entities)):
        entities_file.write(e_inst.premise_entities[i])
        if i < len(e_inst.premise_entities) - 1:
            entities_file.write(", ")
    entities_file.write("\t")
    entities_file.write(e_inst.hypothesis + "\t")
    for i in range(0, len(e_inst.hypothesis_entities)):
        entities_file.write(e_inst.hypothesis_entities[i])
        if i < len(e_inst.hypothesis_entities) - 1:
            entities_file.write(", ")
    entities_file.write("\t" + e_inst.label)
    entities_file.write("\n")

    # Write to triples file
    triples_file.write(e_inst.premise + "\t")
    text_expanded_entities = set()
    triples_entities_file.write(e_inst.premise + "\t")
    for triple in e_inst.premise_triples:
        try:
            triples_file.write(
                "<edge-start>" + triple.subject.value + " " + triple.predicate.value + " " + triple.object.value)
            text_expanded_entities.add(
                conceptneturltostring_utf8(triple.subject.value))
            text_expanded_entities.add(
                conceptneturltostring_utf8(triple.object.value))
        except UnicodeEncodeError:
            "Do Nothing"

    for entity in text_expanded_entities:
        triples_entities_file.write(entity + ", ")

    text_expanded_entities = set()
    triples_file.write("\t")
    triples_entities_file.write("\t")
    triples_file.write(e_inst.hypothesis + "\t")
    triples_entities_file.write(e_inst.hypothesis + "\t")
    for triple in e_inst.hypothesis_triples:
        try:
            triples_file.write(
                "<edge-start>" + triple.subject.value + " " + triple.predicate.value + " " + triple.object.value)
            text_expanded_entities.add(
                conceptneturltostring_utf8(triple.subject.value))
            text_expanded_entities.add(
                conceptneturltostring_utf8(triple.object.value))
        except UnicodeEncodeError:
            "Do Nothing"

    for entity in text_expanded_entities:
        triples_entities_file.write(entity + ", ")

    # write labels
    triples_file.write("\t" + e_inst.label + "\n")
    triples_entities_file.write("\t" + e_inst.label + "\n")

    entities_file.close()
    triples_file.close()
    no_entities_file.close()
    triples_entities_file.close()


def process_entailment_files(dir: str, file_format: str = "scitail", kg: str = "conceptnet", isExists: bool = False,
                             factgen: str='onehop'):
    if kg == "conceptnet":
        print("Processing using conceptnet")
        kg_sparql_endpoint = "http://tiresias-3.sl.cloud9.ibm.com:9997/blazegraph/namespace/kb/sparql"
        extractor = ConceptNetEntityExtractorLinker("../data/conceptnet")
    elif kg == "dbpedia":
        print("Processing using dbpedia")
        kg_sparql_endpoint = "http://tiresias-3.sl.cloud9.ibm.com:9999/blazegraph/namespace/kb/sparql"
        extractor = DBpediaSpotlightExtractor(
            "http://tiresias-3.sl.cloud9.ibm.com:9998/rest/annotate")

    sub_directory_name = kg + "_processed"
    if factgen == 'onehop':
        factgenerator = OneHopFactGenerator(
            sparql_endpoint=kg_sparql_endpoint)
    elif factgen == 'twohoppath':
        factgenerator = TwoHopPathBasedFactGenerator(sparql_endpoint=kg_sparql_endpoint)

    #text = "If a substance has a ph value greater than 7,that indicates that it is base."
    #entities, links = extractor.extract_with_entities_returned(text.lower())
    text_to_conceptnet_graph = TextToSparqlGraphGenerator(
        extractor=extractor, fact_generator=factgenerator)
    #triples = text_to_conceptnet_graph.generate_graph(text.lower())

    if not dir.endswith("/"):
        dir = dir.strip() + "/"
    files = []
    file_ext = ''
    for file in os.listdir(dir):
        if file_format == "scitail":
            file_ext = ".tsv"
            if file.endswith(".tsv") and isDataFile(file):
                files.append(os.path.join(dir, file))

        else:
            file_ext = ".txt"
            if file.endswith(".txt") and isDataFile(file):
                files.append(os.path.join(dir, file))

    sentences_file = dir + file_format + "_sentences" + file_ext
    if not os.path.isfile(sentences_file):
        if len(files) != 3:
            print(files)
            raise AssertionError(
                "There has to be three files in the dataset. Found " + str(len(files)) + " files.")

    print("Creating subdirectories in " + dir)
    if not os.path.isdir(dir + sub_directory_name):
        os.mkdir(dir + sub_directory_name)
        os.mkdir(dir + sub_directory_name + "/entities")
        os.mkdir(dir + sub_directory_name + "/triples")
        os.mkdir(dir + sub_directory_name + "/tripleentities")
        os.mkdir(dir + sub_directory_name + "/noentities")
    else:
        if not isExists:
            print("The path already exists -- Please check the files and rerun or use -exists option")
            exit()

    if not os.path.isfile(sentences_file):
        transform_files_to_sentences(files, dir+file_format+"_sentences"+file_ext)

    process_entailment_file_by_sentences(sentences_file, dir + sub_directory_name + "/",
                                text_to_conceptnet_graph, extractor, isExists, file_format, kg)

def transform_files_to_sentences(input_filepaths, output_filepath):
    sentences = set()
    output_file = open(output_filepath, 'w')
    for input_filepath in input_filepaths:
        with open(input_filepath, "r") as scitail_file:
            for line in scitail_file.readlines():
                if file_format == "snli":
                    premise, hypothesis, label = process_snli_format_line(line)
                elif file_format == "scitail":
                    premise, hypothesis, label = process_scitail_line(line)
                elif file_format == "multinli":
                    premise, hypothesis, label = process_multi_nli_format_line(line)
                sentences.add(premise)
                sentences.add(hypothesis)
    for sentence in sentences:
        output_file.write(sentence+"\n")
    output_file.close()
    print("Created sentences file with " + str(len(sentences)) +" unique sentences.")

def conceptneturltostring_utf8(text: str):
    return text.replace("http://conceptnet.io/c/en/", "").replace("_", " ").strip().encode("utf-8", "ignore").decode("utf-8")


def process_entailment_file(filepath: str, output_dir: str,
                            text_to_graph: TextToGraphGenerator, extractor: Extractor,
                            file_format: str = "scitail", kg: str = "conceptnet"):

    if not output_dir.endswith("/"):
        output_dir = output_dir.strip() + "/"

    filename = path_leaf(filepath)

    file_ext = '.tsv'
    if ".txt" in filepath:
        file_ext = '.txt'

    NUM_WORKERS = 1
    with open(filepath, "r") as scitail_file:
        func = partial(preprocess_tsv_line, output_dir, text_to_graph,
                       extractor, filename, file_format, file_ext, kg)
        with multiprocessing.Pool(processes=NUM_WORKERS) as pool:
            results = pool.map_async(func, scitail_file.readlines())
            results.wait()



def process_entailment_file_by_sentences(filepath: str, output_dir: str,
                            text_to_graph: TextToGraphGenerator, extractor: Extractor, isExists,
                            file_format: str = "scitail", kg: str = "conceptnet"):
    if not output_dir.endswith("/"):
        output_dir = output_dir.strip() + "/"

    filename = path_leaf(filepath)

    file_ext = '.tsv'
    if ".txt" in filepath:
        file_ext = '.txt'

    sentences = set()
    processed_sentences = set()
    if isExists:
        with open(output_dir + "entities/" + filename.replace(file_ext, "_" + kg + "_entities.tsv"), "r") as f:
            for line in f.readlines():
                processed_sentences.add(line.split("\t")[0])
            print("Number of sentences already processed: " + str(len(processed_sentences)))

    with open(filepath, "r") as scitail_file:
        for line in scitail_file.readlines():
            line = line.strip("\n")
            if line not in processed_sentences:
                sentences.add(line)

    NUM_WORKERS = 1
    print("Total number of sentences in " + filename + " is "+ str(len(sentences)))
    func = partial(preprocess_sentence, output_dir, text_to_graph,
                       extractor, filename, file_format, file_ext, kg)
    with multiprocessing.Pool(processes=NUM_WORKERS) as pool:
        results = pool.map_async(func, sentences)
        results.wait()

def preprocess_sentence(output_dir: str, text_to_graph: TextToGraphGenerator,
                        extractor: Extractor, filename: str, file_format: str = "scitail",
                        file_ext: str = ".tsv", kg: str = "conceptnet", sentence: str = None):

    entities_file = open(output_dir + "entities/" +
                         filename.replace(file_ext, "_" + kg + "_entities.tsv"), "a")
    triples_file = open(output_dir + "triples/" +
                    filename.replace(file_ext, "_" + kg + "_triples.tsv"), "a")
    no_entities_file = open(output_dir + "noentities/" +
                        filename.replace(file_ext, "_no_" + kg + ".tsv"), "a")
    triples_entities_file = open(output_dir + "tripleentities/" + filename.replace(file_ext, "_" + kg + "_triple_entities.tsv"),
    "a")

    sentence_entities, sentence_links = extractor.extract_with_entities_returned(sentence.lower())
    sentence_triples = text_to_graph.generate_graph(
        sentence.lower())
    # Write to entities file
    if sentence_entities is None or len(sentence_entities) == 0:
        no_entities_file.write(sentence + "\n")

    entities_file.write(sentence + "\t")
    for i in range(0, len(sentence_entities)):
        entities_file.write(sentence_entities[i])
        if i < len(sentence_entities) - 1:
            entities_file.write(", ")
    entities_file.write("\n")

    # Write to triples file
    triples_file.write(sentence + "\t")
    text_expanded_entities = set()
    triples_entities_file.write(sentence + "\t")
    for triple in sentence_triples:
        try:
            triples_file.write(
                "<edge-start>" + triple.subject.value + " " + triple.predicate.value + " " + triple.object.value)
            subject = triple.subject.value
            object = triple.object.value
            if kg == "conceptnet":
                subject = conceptneturltostring_utf8(triple.subject.value)
                object = conceptneturltostring_utf8(triple.object.value)
            text_expanded_entities.add(
                subject)
            text_expanded_entities.add(
                object)
        except UnicodeEncodeError:
            print("Unicodeerror found but ignored for: " + sentence)
            "Do Nothing"
    triples_file.write("\n")

    for entity in text_expanded_entities:
        triples_entities_file.write(entity + ", ")
    triples_entities_file.write("\n")
    # write labels
    entities_file.close()
    triples_file.close()
    no_entities_file.close()
    triples_entities_file.close()

def preprocess_tsv_line(output_dir: str, text_to_graph: TextToGraphGenerator,
                        extractor: Extractor, filename: str, file_format: str = "scitail",
                        file_ext: str = ".tsv", kg: str = "conceptnet", line: str = None):
    entities_file = open(output_dir + "entities/" +
                         filename.replace(file_ext, "_" + kg + "_entities.tsv"), "a")
    triples_file = open(output_dir + "triples/" +
                        filename.replace(file_ext, "_" + kg + "_triples.tsv"), "a")
    no_entities_file = open(output_dir + "noentities/" +
                            filename.replace(file_ext, "_no_" + kg + ".tsv"), "a")
    triples_entities_file = open(output_dir + "tripleentities/" + filename.replace(file_ext, "_" + kg + "_triple_entities.tsv"),
                                 "a")

    if file_format == "snli":
        premise, hypothesis, label = process_snli_format_line(line)
    elif file_format == "scitail":
        premise, hypothesis, label = process_scitail_line(line)
    elif file_format == "multinli":
        premise, hypothesis, label = process_multi_nli_format_line(line)

    e_inst = entailment_instance(premise, hypothesis, label)

    if kg == "conceptnet":
        print("Processing line: " + line)
        e_inst.premise_entities, premise_links = extractor.extract_with_entities_returned(
            e_inst.premise.lower())
        print("Entities: ")
        print(len(e_inst.premise_entities))
        e_inst.premise_triples = text_to_graph.generate_graph(
            e_inst.premise.lower())
        print("Triples: ")
        print(len(e_inst.premise_triples))

        e_inst.hypothesis_entities, hypothesis_links = extractor.extract_with_entities_returned(
            e_inst.hypothesis.lower())
        e_inst.hypothesis_triples = text_to_graph.generate_graph(
            e_inst.hypothesis.lower())

        # Write to entities file
        if e_inst.premise_entities is None or len(e_inst.premise_entities) == 0:
            no_entities_file.write("P " + premise + "\t" + hypothesis + label + "\n")
            return
        if e_inst.hypothesis_entities is None or len(e_inst.hypothesis_entities) == 0:
            no_entities_file.write("H " + premise + "\t" + hypothesis + "\t" + label + "\n")
            return

        entities_file.write(e_inst.premise + "\t")
        for i in range(0, len(e_inst.premise_entities)):
            entities_file.write(e_inst.premise_entities[i])
            if i < len(e_inst.premise_entities) - 1:
                entities_file.write(", ")
        entities_file.write("\t")
        entities_file.write(e_inst.hypothesis + "\t")
        for i in range(0, len(e_inst.hypothesis_entities)):
            entities_file.write(e_inst.hypothesis_entities[i])
            if i < len(e_inst.hypothesis_entities) - 1:
                entities_file.write(", ")
        entities_file.write("\t" + label)
        entities_file.write("\n")

        # Write to triples file
        triples_file.write(e_inst.premise + "\t")
        text_expanded_entities = set()
        triples_entities_file.write(e_inst.premise + "\t")
        for triple in e_inst.premise_triples:
            try:
                triples_file.write(
                    "<edge-start>" + triple.subject.value + " " + triple.predicate.value + " " + triple.object.value)
                text_expanded_entities.add(
                    conceptneturltostring_utf8(triple.subject.value))
                text_expanded_entities.add(
                    conceptneturltostring_utf8(triple.object.value))
            except UnicodeEncodeError:
                "Do Nothing"

        for entity in text_expanded_entities:
            triples_entities_file.write(entity + ", ")

        text_expanded_entities = set()
        triples_file.write("\t")
        triples_entities_file.write("\t")
        triples_file.write(e_inst.hypothesis + "\t")
        triples_entities_file.write(e_inst.hypothesis + "\t")
        for triple in e_inst.hypothesis_triples:
            try:
                triples_file.write(
                    "<edge-start>" + triple.subject.value + " " + triple.predicate.value + " " + triple.object.value)
                text_expanded_entities.add(
                    conceptneturltostring_utf8(triple.subject.value))
                text_expanded_entities.add(
                    conceptneturltostring_utf8(triple.object.value))
            except UnicodeEncodeError:
                "Do Nothing"

        for entity in text_expanded_entities:
            triples_entities_file.write(entity + ", ")

        # write labels
        triples_file.write("\t" + label + "\n")
        triples_entities_file.write("\t" + label + "\n")

        entities_file.close()
        triples_file.close()
        no_entities_file.close()
        triples_entities_file.close()

    elif kg == "dbpedia":
        print("Processing line: " + line)
        e_inst.premise_entities = extractor.extract(e_inst.premise)
        print(e_inst.premise_entities)
        e_inst.premise_triples = text_to_graph.generate_graph_from_entities(
            e_inst.premise_entities)

        e_inst.hypothesis_entities = extractor.extract(e_inst.hypothesis)
        e_inst.hypothesis_triples = text_to_graph.generate_graph_from_entities(
            e_inst.hypothesis_entities)

        # Write to entities file
        if e_inst.premise_entities is None or len(e_inst.premise_entities) == 0:
            no_entities_file.write("P " + premise)
            return
        if e_inst.hypothesis_entities is None or len(e_inst.hypothesis_entities) == 0:
            no_entities_file.write("H " + hypothesis)
            return

        entities_file.write(e_inst.premise + "\t")
        for i in range(0, len(e_inst.premise_entities)):
            entities_file.write(e_inst.premise_entities[i])
            if i < len(e_inst.premise_entities) - 1:
                entities_file.write(", ")

        entities_file.write("\t")
        entities_file.write(e_inst.hypothesis + "\t")
        for i in range(0, len(e_inst.hypothesis_entities)):
            entities_file.write(e_inst.hypothesis_entities[i])
            if i < len(e_inst.hypothesis_entities) - 1:
                entities_file.write(", ")
        entities_file.write("\n")

        # Write to triples file
        triples_file.write(e_inst.premise + "\t")
        text_expanded_entities = set()
        triples_entities_file.write(e_inst.premise + "\t")
        for triple in e_inst.premise_triples:
            try:
                triples_file.write(
                    "<edge-start>" + triple.subject.value + " " + triple.predicate.value + " " + triple.object.value)
                text_expanded_entities.add(triple.subject.value)
                text_expanded_entities.add(triple.object.value)
            except UnicodeEncodeError:
                "Do Nothing"

        for entity in text_expanded_entities:
            triples_entities_file.write(entity + ", ")

        text_expanded_entities = set()
        triples_file.write("\t")
        triples_entities_file.write("\t")
        triples_file.write(e_inst.hypothesis + "\t")
        triples_entities_file.write(e_inst.hypothesis + "\t")
        for triple in e_inst.hypothesis_triples:
            try:
                triples_file.write(
                    "<edge-start>" + triple.subject.value + " " + triple.predicate.value + " " + triple.object.value)
                text_expanded_entities.add(triple.subject.value)
                text_expanded_entities.add(triple.object.value)
            except UnicodeEncodeError:
                "Do Nothing"

        for entity in text_expanded_entities:
            triples_entities_file.write(entity + ", ")
        triples_file.write("\n")
        triples_entities_file.write("\n")

        entities_file.close()
        triples_file.close()
        no_entities_file.close()
        triples_entities_file.close()


#
# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument(
#         '-i', help='Directory where the files of the entailment dataset resides')
#     parser.add_argument(
#         '-f', help='Format of the files -- snli/scitail/multinli', default="scitail")
#     parser.add_argument(
#         '-g', help='knowledge graph -- conceptnet/dbpedia', default="conceptnet")
#     parser.add_argument('-k', help='fact generator for -- onehop/twohoppath', default='twohoppath')
#     parser.add_argument("-e", help='file exisits', default=False)
#     args = parser.parse_args()
#     input_dir = args.i
#     file_format = args.f
#     isExists = args.e
#     factgen = args.k
#     kg = args.g
#     print("Processing files in " + input_dir)
#     process_entailment_files(input_dir, file_format, kg, factgen=factgen, isExists=isExists)
#     print("Done processing files in " + input_dir)

if __name__ == "__main__":
    entity_file = "/Users/kapanipa/Documents/workspace/reasoning-sciq/ureqa-kg-based-text-entailment/entailment/data/nli_datasets/SciTailV1/tsv_format/conceptnet_processed_twohop/entities/scitail_sentences_conceptnet_entities.tsv"
    entailment_file = "/Users/kapanipa/Documents/workspace/reasoning-sciq/ureqa-kg-based-text-entailment/entailment/data/nli_datasets/SciTailV1/tsv_format/scitail_1.0_train.tsv"
    output_dir = "/Users/kapanipa/Documents/workspace/reasoning-sciq/ureqa-kg-based-text-entailment/entailment/data/nli_datasets/SciTailV1/tsv_format/conceptnet_processed_twohopinter/"
    kg_sparql_endpoint = "http://tiresias-3.sl.cloud9.ibm.com:9997/blazegraph/namespace/kb/sparql"
    process_entity_files(entity_file, entailment_file, output_dir, "scitail", kg_sparql_endpoint)
# def process_entailment_file_for_conceptnet_mt(filepath: str, output_dir: str,
#                                            text_to_graph: TextToGraphGenerator, extractor: Extractor, file_format: str = "scitail"):
#
#     if not output_dir.endswith("/"):
#         output_dir = output_dir.strip()+"/"
#
#     filename = path_leaf(filepath)
#
#     file_ext = '.tsv'
#     if ".txt" in filepath:
#         file_ext = '.txt'
#
#     NUM_WORKERS = 4
#     with open(filepath, "r") as scitail_file:
#         func = partial(preprocess_tsv_line, output_dir, text_to_graph, extractor, filename, file_format, file_ext)
#         with concurrent.futures.ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
#             futures = {executor.submit(func, line) for line in scitail_file.readlines()}
#             concurrent.futures.wait(futures)
