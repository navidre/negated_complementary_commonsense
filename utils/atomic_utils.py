# fact_to_prompt method is an updated form from: https://github.com/allenai/comet-atomic-2020/blob/f5fe091810539707dfbc962f6a6c04f557575d5b/models/gpt2_zeroshot/gpt2-zeroshot.py

import inflect
inflection_engine = inflect.engine()

import spacy
nlp = spacy.load("en_core_web_sm")

import argparse, json

import pandas as pd

# ATOMIC-2020 predicates
atomic_physical_preds = ['ObjectUse', 'AtLocation', 'MadeUpOf', 'HasProperty', 'CapableOf', 'Desires', 'NotDesires']
atomic_event_preds = ['isAfter', 'HasSubEvent', 'isBefore', 'HinderedBy','Causes', 'xReason', 'isFilledBy']
atomic_social_preds = ['xNeed', 'xAttr', 'xEffect', 'xReact', 'xWant', 'xIntent', 'oEffect', 'oReact', 'oWant']
atomic_preds = atomic_physical_preds + atomic_event_preds + atomic_social_preds

def article(word):
    # TODO: Eliminate articles for plural nouns for now
    # return "an" if word[0] in ['a', 'e', 'i', 'o', 'u'] else "a"
    return ''

def vp_present_participle(phrase):
    doc = nlp(phrase)
    return ' '.join([
        inflection_engine.present_participle(token.text) if token.pos_ == "VERB" and token.tag_ != "VGG" else token.text
        for token in doc
    ])

def posessive(word):
    if inflection_engine.singular_noun(word) is False:
        return "have"
    else:
        return "has"

def verbalize_subject_predicate(kg, triple):
    head = triple['head']
    relation = triple['relation']
    tail = triple['tail'][0]

    if kg == "conceptnet" or kg == "transomcs":
        if relation == "AtLocation":
            prompt = "You are likely to find {} {} in {} ".format(
                article(head), head, article(tail)
            )
        elif relation == "CapableOf":
            prompt = "{} can ".format(head)
        elif relation == "CausesDesire":
            prompt = "{} would make you want to ".format(head)
        elif relation == "Causes":
            prompt = "Sometimes {} causes ".format(head)
        elif relation == "CreatedBy":
            prompt = "{} is created by".format(head)
        elif relation == "Desires":
            prompt = "{} {} desires".format(article(head), head)
        elif relation == "HasA":
            prompt = "{} {} ".format(head, posessive(head))
        elif relation == "HasPrerequisite":
            prompt = "{} requires ".format(vp_present_participle(head))
        elif relation == "HasProperty":
            prompt = "{} is ".format(head)
        elif relation == "MotivatedByGoal":
            prompt = "You would {} because you are ".format(head)
        elif relation == "ReceivesAction":
            prompt = "{} can be ".format(head)
        elif relation == "UsedFor":
            prompt = "{} {} is for ".format(article(head).upper(), head)
        elif relation == "HasFirstSubevent" or relation == "HasSubevent" or relation == "HasLastSubevent":
            prompt = "While {}, you would ".format(vp_present_participle(head))
        elif relation == "InheritsFrom":
            prompt = "{} inherits from".format(head)
        elif relation == "PartOf":
            prompt = "{} {} is a part of {} ".format(article(head).upper(), head, article(tail))
        elif relation == "IsA":
            prompt = "{} is {} ".format(head, article(tail))
        elif relation == "InstanceOf":
            prompt = "{} is an instance of".format(head)
        elif relation == "MadeOf":
            prompt = "{} is made of".format(head)
        elif relation == "DefinedAs":
            prompt = "{} is defined as ".format(head)
        elif relation == "NotCapableOf":
            prompt = "{} is not capable of".format(head)
        elif relation == "NotDesires":
            prompt = "{} {} does not desire".format(article(head), head)
        elif relation == "NotHasA":
            prompt = "{} does not have a".format(head)
        elif relation == "NotHasProperty" or relation == "NotIsA":
            prompt = "{} is not".format(head)
        elif relation == "NotMadeOf":
            prompt = "{} is not made of".format(head)
        elif relation == "SymbolOf":
            prompt = "{} is a symbol of".format(head)
        else:
            raise Exception(relation)
    elif kg == "atomic" or kg == "atomic2020":
        if relation == "AtLocation":
            prompt = "You are likely to find {} {} in {} ".format(
                article(head), head, article(tail)
            )
        if relation == "NotAtLocation":
            prompt = "You are not likely to find {} {} in {} ".format(
                article(head), head, article(tail)
            )

        elif relation == "CapableOf":
            prompt = "{} can ".format(head)
        elif relation == "NotCapableOf":
            prompt = "{} cannot ".format(head)

        elif relation == "Causes":
            prompt = "Sometimes {} causes ".format(head)
        elif relation == "NotCauses":
            prompt = "{} does not cause ".format(head)

        elif relation == "Desires":
            prompt = "{} {} desires".format(article(head), head)
        elif relation == "NotDesires":
            prompt = "{} {} does not desire".format(article(head), head)

        elif relation == "HasProperty":
            prompt = "{} is ".format(head)
        elif relation == "NotHasProperty":
            prompt = "{} is not ".format(head)

        elif relation == "HasSubEvent":
            prompt = "While {}, you would ".format(vp_present_participle(head))
        elif relation == "NotHasSubEvent":
            prompt = "While {}, you would not ".format(vp_present_participle(head))

        elif relation == "HinderedBy":
            prompt = "{}. This would not happen if".format(head)
        elif relation == "NotHinderedBy":
            prompt = "{}. This happens even if".format(head)

        elif relation == "MadeUpOf":
            prompt = "{} {} contains".format(article(head), head)
        elif relation == "NotMadeUpOf":
            prompt = "{} {} does not contain".format(article(head), head)

        elif relation == "ObjectUse":
            prompt = "{} {} can be used for".format(article(head), head)
        elif relation == "NotObjectUse":
            prompt = "{} {} cannot be used for".format(article(head), head)

        elif relation == "isAfter":
            prompt = "{}. Before that, ".format(head)
        elif relation == "NotisAfter":
            prompt = "{}. Before that, it is not needed that ".format(head)

        elif relation == "isBefore":
            prompt = "{}. After that, ".format(head)
        elif relation == "NotisBefore":
            prompt = "{}. After that, does not usually ".format(head)

        elif relation == "isFilledBy":
            prompt = "{} is filled by".format(head) #TODO
        elif relation == "NotisFilledBy":
            prompt = "{} is not filled by".format(head) #TODO

        elif relation == "oEffect":
            prompt = "{}. The effect on others will be".format(head)
        elif relation == "NotoEffect":
            prompt = "{}. The effect on others will not be".format(head)

        elif relation == "oReact":
            prompt = "{}. As a result, others feel".format(head)
        elif relation == "NotoReact":
            prompt = "{}. As a result, others do not feel".format(head)

        elif relation == "oWant":
            prompt = "{}. After, others will want to".format(head)
        elif relation == "NotoWant":
            prompt = "{}. After, others will not want to".format(head)

        elif relation == "xAttr":
            prompt = "{}. PersonX is".format(head)
        elif relation == "NotxAttr":
            prompt = "{}. PersonX is not".format(head)

        elif relation == "xEffect":
            prompt = "{}. The effect on PersonX will be".format(head)
        elif relation == "NotxEffect":
            prompt = "{}. The effect on PersonX will not be".format(head)

        elif relation == "xIntent":
            prompt = "{}. PersonX did this to".format(head)
        elif relation == "NotxIntent":
            prompt = "{}. PersonX did this not to".format(head)

        elif relation == "xNeed":
            prompt = "{}. Before, PersonX needs to".format(head)
        elif relation == "NotxNeed":
            prompt = "{}. Before, PersonX does not needs to".format(head)

        elif relation == "xReact":
            prompt = "{}. PersonX will be".format(head)
        elif relation == "NotxReact":
            prompt = "{}. PersonX will not be".format(head)

        elif relation == "xReason":
            prompt = "{}. PersonX did this because".format(head)
        elif relation == "NotxReason":
            prompt = "{}. PersonX did this not because".format(head)

        elif relation == "xWant":
            prompt = "{}. After, PersonX will want to".format(head)
        elif relation == "NotxWant":
            prompt = "{}. After, PersonX will not want to".format(head)
    else:
        raise Exception("Invalid KG")

    return prompt.strip()


if __name__ == "__main__":
    print('Loading spacy ...')
    parser = argparse.ArgumentParser()
    parser.add_argument("--kg", type=str, default="atomic2020", choices=["conceptnet", "transomcs", "atomic", "atomic2020", "wpkg", "wpkg_expanded"])
    parser.add_argument("--input", type=str, default="data/atomic2020/test.tsv")
    parser.add_argument("--output", type=str, default="experiments/test_experiment/sample_prompt.txt")
    args = parser.parse_args()


    # Load args.input as a pandas dataframe
    df = pd.read_csv(args.input, sep="\t", header=None, names=["head", "relation", "tail"])
    row = df.sample()
    # Pandas row to dict
    row_dict = row.to_dict(orient="records")[0]
    # Generate prompt based on row_dict
    verbalized_subject_predicate = verbalize_subject_predicate(args.kg, row_dict)

    print(f'row_dict: {row_dict}')
    print(f'verbalized_subject_predicate: {verbalized_subject_predicate}')