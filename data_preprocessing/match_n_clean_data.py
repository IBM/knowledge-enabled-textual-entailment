
# coding: utf-8

# # Match & Clean Data
#

# In[4]:


from pathlib import Path

#task = 'entities'
task = 'triples'

DATA_ROOT = Path("../data/nli_datasets/SciTailV1/tsv_format/")
KG_DATA_ROOT = Path(
    "../data/nli_datasets/SciTailV1/tsv_format/conceptnet_processed/" + task)
PARSED_ROOT = Path(
    "../data/nli_datasets/SciTailV1/tsv_format/conceptnet_processed/" + task + "/parsed")

if not PARSED_ROOT.exists():
    PARSED_ROOT.mkdir()


from collections import defaultdict

divisions = ["train", "dev", "test"]
formats = [task]
for division in divisions:
    # map each sentence to its entities / triples
    premise_map = defaultdict(str)
    hypothesis_map = defaultdict(str)
    for format_ in formats:
        if format_ == "entities":
            with open(KG_DATA_ROOT / f"scitail_1.0_{division}_conceptnet_{format_}.tsv", encoding='utf-8', errors='ignore') as f:
                for line in f:
                    premise, p_content, hypothesis, h_content = line.strip().split('\t')
                    premise_map[premise] = p_content
                    hypothesis_map[hypothesis] = h_content
        elif format_ == "triples":
            with open(KG_DATA_ROOT / f"scitail_1.0_{division}_conceptnet_{format_}.tsv", encoding='utf-8', errors='ignore') as f:
                for line in f:
                    # One of the lines is not parsing, hence a hack
                    try:
                        premise, p_content, hypothesis, h_content = line.strip().split('\t')
                    except ValueError:
                        continue
                    h_triples = h_content.lstrip().strip().split("<edge-start>")
                    p_triples = p_content.lstrip().strip().split("<edge-start>")
                    h_content_entities_only = set()
                    p_content_entities_only = set()

                    for triple in h_triples:
                        if not triple:
                            # This happens only in the beginning
                            continue
                        subj, pred, obj = triple.split(" ")
                        subj = subj.replace(
                            "http://conceptnet.io/c/en/", "").replace("_", " ").strip()
                        obj = obj.replace(
                            "http://conceptnet.io/c/en/", "").replace("_", " ").strip()
                        try:
                            h_content_entities_only.add(subj)
                            h_content_entities_only.add(obj)
                        except UnicodeEncodeError:
                            continue
                    for triple in p_triples:
                        if not triple:
                            # This happens only in the beginning
                            continue
                        subj, pred, obj = triple.split(" ")
                        subj = subj.replace(
                            "http://conceptnet.io/c/en/", "").replace("_", " ").strip()
                        obj = obj.replace(
                            "http://conceptnet.io/c/en/", "").replace("_", " ").strip()
                        try:
                            p_content_entities_only.add(subj)
                            p_content_entities_only.add(obj)
                        except UnicodeEncodeError:
                            continue
                    premise_map[premise] = ", ".join(p_content_entities_only)
                    hypothesis_map[hypothesis] = ", ".join(
                        h_content_entities_only)

        with open(DATA_ROOT / f"scitail_1.0_{division}.tsv", encoding='utf-8', errors='ignore') as orig:
            with open(PARSED_ROOT / f"scitail_1.0_{division}_conceptnet_{format_}.tsv", 'w', encoding='utf-8', errors='ignore') as parsed:
                with open(PARSED_ROOT / f"scitail_1.0_{division}_conceptnet_no_{format_}.tsv", 'w', encoding='utf-8', errors='ignore') as failed:
                    for line in orig:
                        premise, hypothesis, label = line.strip().split('\t')
                        p_content = premise_map[premise]
                        h_content = hypothesis_map[hypothesis]
                        out_file = failed if p_content == '' or h_content == '' else parsed
                        print(f"{premise}\t{p_content}\t{hypothesis}\t{h_content}\t{label}",
                              file=out_file)
