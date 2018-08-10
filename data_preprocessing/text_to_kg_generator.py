import nltk
import spacy
from spacy_lookup import Entity
from entailment.knowledge_graph.fact_generator import OneHopFactGenerator, FactGenerator
import os
import string
from nltk.corpus import stopwords
from datetime import datetime
from nltk import ngrams
import time
import requests


class Extractor():
    def extract(self, text: str) -> list:
        raise NotImplementedError

class DBpediaSpotlightExtractor:
    """
        Spotlight extractor to extract entities and link them to dbpedia.
    """
    def __init__(self, address: str, args: dict=None):
        print("Initializing DBpedia Spotlight.")
        self.address = address
        if args is None:
            self.args = {}
            self.args["confidence"] = 0.0
            self.args["support"] = 0
            self.args["spotter"] = "Default"
            self.args["disambiguator"] = "Default"
            self.args["policy"] = "whitelist"
            self.args["headers"] = None
            self.args["format"] = "list"
        self.extract("temporary hacks to see whats wrong with Barack Obama")


    def extract(self, text: str):
        #print("Extracting: " + text)
        params = {'confidence': self.args["confidence"], 'support': self.args["support"],
                  'spotter': self.args["spotter"], 'disambiguator': self.args["disambiguator"],
               'policy': self.args["policy"], 'text': text}
        if self.args["format"] == "json" or self.args["format"] =="list":
            req_headers = {'accept': 'application/json'}
        elif self.args["format"] == "xml":
            req_headers = {'accept': 'application/xml'}
        req_headers.update(self.args["headers"] or {})

        response = requests.post(self.address, data=params, headers=req_headers)

        # Check if the response is 200 or not
        if response.status_code != requests.codes.ok:
            # Every http code besides 200 shall raise an exception.
            print(str(response.status_code) +": "+ text)
            return None

        response_dictionary = response.json()

        if 'Resources' not in response_dictionary:
            return None

        if format == "json" :
            return response_dictionary['Resources']

        return [entity_resource['@URI'] for entity_resource in response_dictionary['Resources']]


class SpacyDictBasedEntityExtractor(Extractor):
    def __init__(self, entities: list, remove_stopwords: bool = True):
        self.spacyload = spacy.load('en_core_web_sm')
        self.ext = Entity(keywords_list=entities)
        self.spacyload.add_pipe(self.ext, last=True)
        self.error_text = 0
        self.no_stopwords = remove_stopwords
        self.stopwords = set(stopwords.words('english'))

    def extract(self, text: str) -> list:
        """
        TODO: Extend this to send annotated text which comprises all the info in doc of spacy lookup
        """
        candidates = []
        try:
            # Hacking here to make sure spacy works
            doc = self.spacyload(self._fix_for_spacy(text))
            entity_tuples = doc._.entities
            for surface, start, entity in entity_tuples:
                if self.no_stopwords:
                    if entity not in self.stopwords:
                        candidates.append(entity)
                    else:
                        continue
                else:
                    candidates.append(entity)
        except Exception:
            print("Spacy Not able to extract entities Warning: " + text)
            "DO NOTHING"
        return candidates

    def _fix_for_spacy(self, text: str):
        """
        Worst hack ever. But for now and for it to work :D
        :param text:
        :return:
        """
        replace_punctuation = text.maketrans(string.punctuation, ' ' * len(string.punctuation))
        text = text.translate(replace_punctuation).strip()
        return text

class NLTKBasedEntityExtractor(Extractor):
    def __init__(self, entities: list, remove_stopwords: bool = True, ngram: int = 3, max_substring: bool=True):
        self.entities = set(entities)
        self.no_stopwords = remove_stopwords
        self.n = ngram
        self.stopwords = set(stopwords.words('english'))
        self.only_max_substring = max_substring

    def extract(self, text: str) -> list:
        """
        This is Max substring match
        """
        candidates = []
        text_ngrams = []
        text = self.replace_punctuations(text)
        # creates n grams
        for i in range(self.n, 0, -1):
            grams = ngrams(text.split(), i)
            n_gram = ''
            for gram in grams:
                max_substring = True
                gram_con = ''
                for term in gram:
                    gram_con += term +' '
                gram_con = gram_con.strip()
                if gram_con in self.entities:
                    if self.only_max_substring:
                        for candidate in candidates:
                            if gram_con in candidate:
                                max_substring = False

                        if max_substring:
                            if self.no_stopwords:
                                if gram_con not in self.stopwords:
                                        candidates.append(gram_con)
                            else:
                                candidates.append(gram_con)
                    else:
                        if self.no_stopwords:
                            if n_gram not in self.stopwords:
                                candidates.append(gram_con)
                        else:
                            candidates.append(gram_con)

        return candidates

    def replace_punctuations(self, text: str):
        """
        Worst hack ever. But for now and for it to work :D
        :param text:
        :return:
        """
        replace_punctuation = text.maketrans(string.punctuation, ' ' * len(string.punctuation))
        text = text.translate(replace_punctuation).strip()
        return text


class ConceptNetEntityExtractorLinker(Extractor):
    def __init__(self, dir_path: str = "../data/concepnet/", extractor: str = "max_substring"):
        if not dir_path.endswith("/"):
            dir_path = dir_path +"/"
        self.entities_file = dir_path+"conceptnet_entities.txt"
        assert os.path.exists(self.entities_file)
        self.entities_links_file = dir_path+"conceptnet_entities_links.txt"
        assert os.path.exists(self.entities_links_file)
        self.entities_links = {}
        with open(self.entities_links_file) as conceptnet_entities_links_file:
            for line in conceptnet_entities_links_file.readlines():
                entity_info = line.split("\t")
                self.entities_links[entity_info[0].strip()] = entity_info[1].strip()
        self.entity_linker = EntityLinking(self.entities_links, extractor)

    def extract(self, text: str):
        # Assuming default extractor here
        return self.entity_linker.extract(text)

    def extract_with_entities_returned(self, text: str):
        # Assuming default extractor here
        return self.entity_linker.extract_with_entities_returned(text)

    def link_entity(self, entity:str):
        return self.entity_linker.link_entities([entity])


class EntityLinking(Extractor):
    "A very naive entity linker which returns all links without disambiguation"
    def __init__(self, entities_dict: dict, entity_extractor: str = "max_substring"):
        self.inverted_index = self._inverted_entity_index(entities_dict)
        self.entities_text = list(self.inverted_index.keys())
        if entity_extractor == "spacy":
            self.extractor = SpacyDictBasedEntityExtractor(self.entities_text)
        else:
            self.extractor = NLTKBasedEntityExtractor(self.entities_text)

    def _inverted_entity_index(self, entities_dict, ignore_case=True):
        inverted_index = {}
        for entity_url in entities_dict:
            urls = []
            entity_text = entities_dict[entity_url]
            if ignore_case:
                entity_text = entity_text.lower()
            if entity_text in inverted_index:
                urls = inverted_index[entity_text]
            urls.append(entity_url)
            inverted_index[entity_text] = urls
        return inverted_index

    def extract(self, text: str):
        entities = self.extractor.extract(text)
        return self.link_entities(entities)

    def extract_with_entities_returned(self, text: str):
        entities = self.extractor.extract(text)
        return entities, self.link_entities(entities)

    def link_entities(self, entities: list):
        entities_linked = []
        for entity in entities:
            link = self.inverted_index[entity]
            entities_linked.extend(link)
        return entities_linked


class TextToGraphGenerator():
    def generate_graph(self, text: str):
        raise NotImplementedError

    def generate_graph_from_entities(self, concepts: list):
        raise NotImplementedError


class TextToSparqlGraphGenerator(TextToGraphGenerator):
    def __init__(self, extractor: Extractor, fact_generator: FactGenerator, sparql_endpoint: str = None):
        self.concept_extractor = extractor
        self.fact_generator = fact_generator
        self.sparql_endpoint = sparql_endpoint

    def generate_graph(self, text: str):
        concepts = self.concept_extractor.extract(text)
        #print(concepts)
        triples = self.fact_generator.generate_facts(concepts, self.sparql_endpoint)
        #print("Length")
        #print(len(triples))
        return triples

    def generate_graph_from_entities(self, concepts: list):
        return self.fact_generator.generate_facts(concepts, self.sparql_endpoint)

def test():
    text = "If a substance has a ph value greater than 7,that indicates that it is base."
    extractor = ConceptNetEntityExtractorLinker("/Users/kapanipa/Documents/workspace/reasoning-sciq/ureqa-kg-based-text-entailment/entailment/data/conceptnet", extractor="max_substring")
    start = time.time()
    entities, links = extractor.extract_with_entities_returned(text.lower())
    print(time.time() - start)
    onehopfactgen = OneHopFactGenerator(sparql_endpoint="http://tiresias-3.sl.cloud9.ibm.com:9997/blazegraph/namespace/kb/sparql")
    text_to_conceptnet_graph = TextToSparqlGraphGenerator(extractor, onehopfactgen)
    start = time.time()
    triples = text_to_conceptnet_graph.generate_graph(text.lower())
    print(time.time() - start)
    print(links)
    print(len(triples))
    print(len(entities))

def testspacyextractor():
    entities = []
    with open("/Users/kapanipa/Documents/workspace/reasoning-sciq/ureqa-kg-based-text-entailment/entailment/data/conceptnet/conceptnet_entities.txt") as entities_file:
        for line in entities_file.readlines():
            entities.append(line.strip())

    # extractor = SpacyDictBasedEntityExtractor(entities)
    # start=time.time()
    # entities = extractor.extract(str("If a substance has a ph value greater than 7,that indicates that it is base.").lower())
    # print(len(entities))
    # print(entities)
    # print(time.time() - start)
    # start = time.time()
    # entities = extractor.extract(str("Based on the list provided of the uses of substances 1-7, estimate the pH of each unknown and record the number in the data table in the estimated pH column.").lower())
    # print(len(entities))
    # print(entities)
    # print(time.time() - start)
    # start = time.time()

    extractor = NLTKBasedEntityExtractor(entities, max_substring=True)
    start = time.time()
    entities = extractor.extract(
        str("If a substance has a ph value greater than 7,that indicates that it is base.").lower())
    #print(len(entities))
    #print(entities)
    print(time.time() - start)
    extractor = SpacyDictBasedEntityExtractor(entities)
    start = time.time()
    entities = extractor.extract(str(
        "Based on the list provided of the uses of substances 1-7, estimate the pH of each unknown and record the number in the data table in the estimated pH column.").lower())
    print(len(entities))
    print(entities)
    print(time.time() - start)

#test()