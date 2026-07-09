import json, glob, os, random, re
import langid
fem = json.load(open("fem.json"))
from names import *
from flagged_words import *
# Use a pipeline as a high-level helper
#from transformers import pipeline

#pipe = pipeline("summarization", model="Falconsai/text_summarization")

#pipe = pipeline("summarization", model="Babelscape/t5-base-summarization-claim-extractor")


#pipe = pipeline("summarization", model="marianna13/flan-t5-base-summarization")


from collections import Counter

from PIL import Image
def cjk_detect(text):
    """
    Detects if a text contains characters from specific East Asian languages (Chinese, Japanese, Korean, Thai, and Traditional Javanese).

    Args:
        text (str): Text to check.

    Returns:
        str or None: Language code if detected; otherwise, None.
    """
    # chinese
    if re.search("[\u4e00-\u9FFF]", text):
        return "zh"
    # korean
    if re.search("[\uac00-\ud7a3]", text):
        return "ko"
    # japanese
    if re.search("[\u3040-\u30ff]", text):
        return "ja"
    # thai
    if re.search("[\u0E01-\u0E5B]", text):
        return "th"
    # traditional javanese
    if re.search("[\uA980-\uA9DF]", text):
       return "jv_tr"
    return None


def get_ngram(text, window_size=3, lang=""):
    if not lang:
        if cjk_detect(text[:min(len(text), 100)]):
            lang = 'zh'
        else:
            lang = 'en'
    if lang in {"zh", "ja", "ko", "th", "jap"}:
        tokens = text
        ret = [
            "".join(tokens[i : i + window_size])
            for i in range(len(tokens) - window_size)
        ]
        ret = [
        "".join(tokens[i : i + window_size]) for i in range(len(tokens) - window_size)
        ]
        
    else:
        tokens = text.split(" ")
        ret = [
            " ".join(tokens[i : i + window_size]) for i in range(len(tokens) - window_size)
        ]
    return Counter(ret)

def fix_too_much_ngram(text, window_size=3, lang="en", threshold=2, logger=None):
    stopwords =  all_stopwords.get(lang, all_stopwords['en'])
    for word, cnt in get_ngram(text, window_size=3, lang="").items():
        if cnt >= threshold:
            word_arr = word.split()
            if not any(w for w in word_arr if len(w) > 3 and w.lower() not in stopwords and w.lower()[:min(len(w), 4)] not in {'said', 'says', 'sayi', 'menti', 'disc', 'talk', 'desc', 'hear', 'speak',}) :
                continue
            if word not in text:
                if logger: logger.warning(("NGRAM NOT IN TEXT", word, text))
                continue
            i = text.index(word)
            text = text[:i+1]+text[i+1:].replace("and "+word, "").replace("or "+word, "").replace(", "+word, "").replace(" "+word, "")
    text = text.split(" ")
    len_text = len(text)
    for i in range(len_text):
        if i < len_text - 2:
            if text[i] == text[i+1] and text[i] == text[i+2]:
                text[i] = None
    text = " ".join(t for t in text if t is not None)
    return text


def add_punc(text):
    return fix_too_much_ngram(text.replace("it's", ". It's").replace(" sorry", ". Sorry").replace(" um ", ". ").replace(" uh ", ". ").replace(" ugh ", " . ").replace(" uhm ", " . ").replace("Okay", ". ").replace("okay", ". ").replace("yeah", ". ").replace("Yeah", ". ").replace(" hey ", ". Hey").replace(" so ", ". So ").replace(" let's", ". Let's").replace(" and ", ". And ").replace(" I'", ". I'").replace(" I am", ". I am").replace(" I will", ". I will").replace(" An", ". An").replace(" It", ". It").replace(" She", ". She").replace(" He", ". He").replace(" Th", ". Th").replace(". And. ", ". And").replace(" .", ".").replace("..", ".").replace(",.", ".").lstrip(". "))

def find_snac(ogg):
    if  os.path.exists("/mnt/sda/snac1/"+ogg):
        return "/mnt/sda/snac1/"+ogg
    if  os.path.exists("/mnt/sda/snac2/"+ogg):
        return "/mnt/sda/snac2/"+ogg
    if  os.path.exists("/mnt/sda/snac3/"+ogg):
        return "/mnt/sda/snac3/"+ogg
    if  os.path.exists("/mnt/sda/snac4/"+ogg):
        return "/mnt/sda/snac4/"+ogg
    if  os.path.exists("/mnt/sda/snac5/"+ogg):
        return "/mnt/sda/snac5/"+ogg

import base64
from PIL import Image
import io

def encode_to_base64(image_path, size=(100, 75)):
    with Image.open(image_path) as img:
        img = img.convert('RGB')
        if size:
            img = img.resize(size, Image.LANCZOS) # Use LANCZOS for high quality resizing
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG")
        value =  buffer.getvalue()
        encoded_bytes = base64.b64encode(value)
        return size, f"data:image/jpg;base64,"+encoded_bytes.decode('utf-8')

    
def process(l):
    out = []
    data = json.loads(l)
    text = data["text"]
    lang = data["language"]
    id = json.loads(data["metadata"]["params"])["id"]
    transcripts = json.loads(data["metadata"]["params"])["transcripts"]

    prev_is_say = False
    text = text.replace("The image shows a ", "A ").replace("The image is a ", "A ").replace("The image is an ", "An ").replace("A screenshot of a", "A").replace("A screenshot from a", "A").replace(" : : ", " : ").replace(": :", ":").replace(": :", ":").replace(": :", ":").replace("\n : \n", "").replace(" : : ", " : ").replace(": :", ":").replace(": :", ":").replace(": :", ":").replace(":  : ", ": ").replace(".. .. ", ".. ").replace(".. .. ", ".. ").replace(".. .. ", ".. ").replace(".. .. ", ".. ").replace(".. .. ", ".. ")
    text2 = []
    for t in text.split(" "):
        if len(text2) > 2 and text2[-1] == t and text2[-2] == t:
            continue
        if "http" in t:
            t = t.replace(t.strip("~!@#$%^&*()-_+\""), "")
        text2.append(t)
    text = " ".join(text2)                    
    for idx, t0 in enumerate(text.split(">")):
        t0 = t0.split("<")[0]
        for t1 in t0.split("."):
            if len(t1) > 30 and t1 in text:
                idx2 = text.index(t1)
                text = text[:idx2+5]+text[idx2+5:].replace(t1+".", " ")
                text = text[:idx2+5]+text[idx2+5:].replace(t1, " ")
    if lang != "en":            
        for idx, t0 in enumerate(text.split("<|endofsection|>")):
            if idx == 0: continue
            t0 = t0.split(">")[-1]
            t1 = add_punc(t0)

            if len(t0) > 30 and text.count(t0) == 1  and "subscribe" not in t0:
                if t0+"<|endofsection|>" in text:
                    text = text.replace(t0+"<|endofsection|>", "English translation: "+t1+"\n")
                else:
                    text = text.replace(t0, "English translation: "+t1+"\n")                        
    for idx, t0 in enumerate(text.split("<|endofsection|>")):
        if idx == 0: continue
        t0 = t0.split(">")[-1]
        if len(t0) < 30:
            text = text.replace(t0+"<|endofsection|>", "\n")
            text = text.replace("<|endofsection|>"+t0, "\n")
    text = text.replace("<|endofsection|>", "\n")                    
    data["media"] = json.loads(data["media"])
    qa = ""
    done_base64 = ""
    for k, v in data["media"].items():
        answer = text.split(k)[-1].split("<")[0]
        if "image" in k and k in text:
            image_path = "/mnt/sdb/all_seed/"+v.split("/")[-1].replace(".png", "_seed2.jsonl")
            if os.path.exists(image_path):
                try:
                    image_toks = "<see>"+"".join(["<seed_"+str(a)+">" for a in json.load(open(image_path))])+"</see>"
                except:
                    continue
                if answer:
                     text = text.replace("<caption>"+k+answer+"</caption>", k+image_toks+"\n")
                     text = text.replace(k+answer, k+image_toks)                             
                     qa += "\nQ: What did you see at time "+k.split("_")[-1].strip("<>")+"?\nA: "+ answer
                     #print ((qa,text))
                if random.randint(0,10) == 0:
                    image_path = image_path.replace("_seed2.jsonl", ".png",)
                    answer = answer.replace("English translation: ", "").strip()
                    if os.path.exists(image_path):
                        size, encoded_string = encode_to_base64(image_path, random.choice([[100, 100], [100, 75], [50, 40], [75, 60], [200,200]]))
                        if random.randint(0,1):
                            done_base64 = f"\nQ: Imagine this scene:\n{image_toks}\n==\nGive me the base64 encoding for this scene.\nA: "+encoded_string+"\nQ: Describe the image.\nA: "+answer
                        else:
                            done_base64 = f"\nQ: Imagine this scene:\n{answer}\n==\nGive me the base64 encoding for this scene.\nA: "+encoded_string+"\nQ: Now you are looking at the base64 image. What do you see here? "+image_toks+".\nA: "+answer
                        next_dialog = [a["text"] for a in transcripts.values() if k in a["text"]]
                        if next_dialog:
                            done_base64 += f"\nQ: What would the speaker narrating this scene say?\nA: " + add_punc(next_dialog[0].split(">")[-1])
                        data2 = {"text": done_base64.strip(), "metadata": [data["metadata"]]}

                        out.append(data2)
                continue     
            image_path = image_path.replace("_seed2.jsonl", ".png",)
            if os.path.exists(image_path) and random.randint(0,5)==0:
                size, encoded_string = encode_to_base64(image_path, random.choice([[100, 100], [100, 75], [50, 40], [75, 60], [200,200] ]))
                done_base64 = f"\nQ: Imagine this scene:\n{answer}\n==\nGive me the base64 encoding for this scene.\nA: "+encoded_string+"\nQ: Describe the image.\nA: "+answer
                #print (done_base64)
                continue

        if "audio" in k and k in text:
            v = v.split("/")[-1]
            if "[" not in answer:
                ogg_path =  find_snac(v.replace(".ogg", ".clone_speak"))
                if ogg_path:
                    try:
                        snac = json.load(open(ogg_path))
                    except:
                        snac = None
                    if snac:
                        if answer:
                            prev_is_say = True
                            snac =  "<speak>"+"".join(["<snac_"+str(a)+">" for a in snac])+"</speak>"
                            text = text.replace("<transcript>"+k+answer+"</transcript>", k+snac+"\n")
                            text = text.replace(k+answer, k+snac)                                
                            if answer.strip(". ") and answer.strip().lower() != "[music]":
                                if lang == "en":
                                    answer = add_punc(answer)
                                else:
                                    answer = fix_too_much_ngram(answer)
                                qa += "\nQ: What did you say "+("" if random.randint(0,5) != 0 else ("" if lang == "en" else " in "+langs2fullname.get(lang, "")))+" at time "+ k.replace("<audio_", "").strip("<>")+"?\nA: \""+answer+"\""
                                if done_base64:
                                    done_base64 += "\nQ: What do you think of this scene? Say your response.\nA: "+snac
                                    data2 = {"text": done_base64.strip(), "metadata": [data["metadata"]]}
                                    out.append(data2)
                                    done_base64 = ""
                                ogg_path =  find_snac(v.replace(".ogg", ".listen"))
                                listen_snac = ""
                                if ogg_path:
                                    try:
                                        listen_snac = json.load(open(ogg_path))
                                    except:
                                        pass
                                if listen_snac:
                                    listen_snac =  "<listen>"+"".join(["<snac_"+str(a)+">" for a in listen_snac])+"</listen>"
                                    text2 = "Q: Listen to this and tell me what you heard. "+listen_snac+"\nA: "+("" if lang == "en" else " In "+langs2fullname.get(lang, "")+":")+" \""+answer+"\"\nQ: Now repeat what you heard.\nA: "+snac
                                    if lang != "en":
                                        next_dialog = [a["text"] for a in transcripts.values() if k in a["text"]] 
                                        if next_dialog:
                                            text2 += "\nQ: What does this mean in English?\nA:"+ add_punc(next_dialog[0].split(">")[-1])
                                    data2 = {"text": text2, "metadata": [data["metadata"]]}
                                    out.append(data2)
                                continue
            if "[" not in answer and not prev_is_say and id in fem:
                ogg_path =  find_snac(v.replace(".ogg", ".speak"))
                if not ogg_path:
                    ogg_path =  find_snac(v.replace(".ogg", ".listen"))
            else:
                ogg_path =  find_snac(v.replace(".ogg", ".listen"))

            if ogg_path:
                try:
                    snac = json.load(open(ogg_path))
                except:
                    snac = None

                if snac:
                    if answer:
                        act = ogg_path.split(".")[-1]
                        act2 = "say"
                        if act == "listen":
                            act2 = "hear"
                            prev_is_say = False                                                                        
                        else:
                            prev_is_say = True                                    
                        snac =  f"<{act}>"+"".join(["<snac_"+str(a)+">" for a in snac])+f"</{act}>"
                        text = text.replace("<transcript>"+k+answer+"</transcript>", k+snac+"\n")                                
                        text = text.replace(k+answer, k+snac)
                        if answer.strip(". "):
                            if lang == "en":
                                answer = add_punc(answer)
                            else:
                                answer = fix_too_much_ngram(answer)

                            qa += f"\nQ: What did you {act2} "+("" if random.randint(0,5) != 0 else ("" if lang == "en" else " in "+langs2fullname.get(lang, "")))+" at time "+ k.replace("<audio_", "").strip("<>")+"?\nA: \""+answer+"\""
                        if done_base64:
                            if act2 == "say":
                                if "narrating this scene say?\nA: " in done_base64:
                                    done_base64 = done_base64.split("narrating this scene say?\nA: ")[0]+"narrating this scene say?\nA: "+snac
                                    if lang != "en":
                                        done_base64 = done_base64.replace("say?", "say in "+langs2fullname.get(lang, "")+"?")
                            data2 = {"text": done_base64.strip(), "metadata": [data["metadata"]]}
                            out.append(data2)
                            done_base64 = ""
                        if act2 == "say":
                            ogg_path =  find_snac(v.replace(".ogg", ".listen"))
                            listen_snac = ""
                            if ogg_path:
                                try:
                                    listen_snac = json.load(open(ogg_path))
                                except:
                                    pass
                            if listen_snac:
                                listen_snac =  "<listen>"+"".join(["<snac_"+str(a)+">" for a in listen_snac])+"</listen>"
                                text2 = "Q: Listen to this and tell me what you heard. "+listen_snac+"\nA: "+("" if lang == "en" else " In "+langs2fullname.get(lang, "")+":")+" \""+answer+"\"\nQ: Now repeat what you heard.\nA: "+snac
                                if lang != "en":
                                    next_dialog = [a["text"] for a in transcripts.values() if k in a["text"]] 
                                    if next_dialog:
                                        text2 += "\nQ: What does this mean in English?\nA:"+ add_punc(next_dialog[0].split(">")[-1])

                                data2 = {"text": text2, "metadata": [data["metadata"]]}
                                out.append(data2)
                        #if act2 == "say":
                        #    print (qa)
                        continue
            if done_base64.strip():
                data2 = {"text": done_base64.strip(), "metadata": [data["metadata"]]}
                out.append(data2)
                done_base64 = ""

    if done_base64.strip():
        data2 = {"text": done_base64.strip(), "metadata": [data["metadata"]]}
        out.append(data2)
        done_base64 = ""                
    if qa:
        text = text.replace("<video>", "\n").replace("</video>", "\n")
        text = text.strip()
        text = text+"\n===\n"+qa
        del data["text"] 
        data = {"text": text, "metadata":[data["metadata"]]}
        #print (data)
        out.append(data)
    return out
                            
def process_old(l):

    data = json.loads(l)
    data['text'] = data['text'].replace(" : : ", " : ").replace(": :", ":").replace(": :", ":").replace(": :", ":").replace("\n : \n", "").replace(" : : ", " : ").replace(": :", ":").replace(": :", ":").replace(": :", ":").replace(":  : ", ": ").replace(".. .. ", ".. ").replace(".. .. ", ".. ").replace(".. .. ", ".. ").replace(".. .. ", ".. ").replace(".. .. ", ".. ")
    data['metadata'] = json.loads(data['metadata'])
    text_arr = []
    meta_arr = []
    found = False
    for text, metadata in zip(data['text'].split("<|endoftext|>"),  data['metadata']):
        qa = ""
        found = False
        if "<audio" in text:
            prev_speak = False
            for audio in text.split("<audio_"):
                if ">" not in audio:
                    prev_speak = False
                    continue
                ogg_no, audio = audio.split(">",1)
                if len(ogg_no) > 100:
                    prev_speak = False
                    continue
                ogg = metadata['idx']+"_"+ogg_no+".clone_speak"
                ogg_path = find_snac(ogg)
                if ogg_path:# and not os.path.exists(ogg_path+".done"):
                    #open(ogg_path+".done", "w")
                    try:
                        snac = json.load(open(ogg_path))
                    except:
                        snac = None
                    if snac:
                        answer = audio.split("<")[0]
                        snac =  "<speak>"+"".join(["<snac_"+str(a)+">" for a in snac])+"</speak>"
                        text = text.replace("audio_"+ogg_no+">"+answer, "time_"+ogg_no+">"+snac)
                        if answer.strip(". ") and answer.strip().lower() != "[music]":
                            qa += "\nQ: What did you say at time "+ ogg_no+"?\nA: "+answer
                        found = True
                        prev_speak = True
                        if ( "femin" in text[:-200] or  "female" in text[:-200] or "woman" in text[:-200] or "girl" in text[:-200]):
                            ogg = metadata['idx']+"_"+ogg_no+".speak"
                            ogg_path = find_snac(ogg)
                        continue
                ogg = metadata['idx']+"_"+ogg_no+".listen"
                if ( "femin" in text[:-200] or  "female" in text[:-200] or "woman" in text[:-200] or "girl" in text[:-200]):
                    ogg = metadata['idx']+"_"+ogg_no+".speak"
                    ogg_path = find_snac(ogg)
                    if not ogg_path or not prev_speak:
                        ogg = metadata['idx']+"_"+ogg_no+".listen"
                        prev_speak = False
                    else:
                        prev_speak = True
                ogg_path = find_snac(ogg)
                if ogg_path:#  and not os.path.exists(ogg_path+".done"):
                    #open(ogg_path+".done", "w")
                    
                    try:
                        snac = json.load(open(ogg_path))
                        #print ("found", ogg)
                    except:
                        snac = None
                    if snac:
                        answer = audio.split("<")[0]
                        tag = ogg.split(".")[-1]
                        snac =  f"<{tag}>"+"".join(["<snac_"+str(a)+">" for a in snac])+f"</{tag}>"
                        text = text.replace("audio_"+ogg_no+">"+answer, "time_"+ogg_no+">"+snac)
                        if tag == "speak":
                            if answer.strip(". ") and answer.strip().lower() != "[music]":
                                qa += "\nQ: What did you say at time "+ ogg_no+"?\nA: "+answer
                        else:
                            if answer.strip(". "):
                                qa += "\nQ: What did you hear at time "+ ogg_no+"?\nA: "+answer
                        found = True
                        continue
                prev_speak = False
                        
                                
        if found:
            for t0 in text.split("<|endofsection|>"):
                for t1 in  t0.split("<transcript>"):
                    t1 = t1.split("Q:")[0].split("===")[0].strip()
                    for t2 in t1.split("</transcript>"):
                        if "Title:" in t2: continue
                        if "<audio" not in t2:
                            t2 = t2.split(">")[-1]
                        t3 = t2.split(">")[-1]
                        if len(t2) < 50: continue
                        if 'en' == langid.classify(t3[:min(len(t3), 500)])[0]:
                            if "<audio" in t2:
                                if random.randint(0,10) == 0:
                                    answer = add_punc(t2.split(">")[-1])
                                    t4 = t2.split(">")[0].replace("<audio", "<span").strip(">")+">" 
                                    text = text.replace("<transcript>"+t2+"</transcript>",t4)
                                    text = text.replace(t2+"</transcript>", t4)
                                    text = text.replace("<transcript>"+t2, t4)
                                    text = text.replace(t2, t4)
                                    qa += "\nQ: Fill in span "+t2.split(">")[0].split("_")[-1]+"?\nA: "+answer
            text = text.replace("<transcript>", "").replace("<audio", "<transcript").replace("</listen></transcript>", "</listen>").replace("</speak></transcript>", "</speak>")
            text_arr.append(text+qa)
            meta_arr.append(metadata)
    if not text_arr: return
    data['text'] = "<|endoftext|>".join(text_arr)
    data['meta_arr'] = json.dumps(meta_arr)
    return data
#print (data)


import multiprocessing
import tqdm
idx1 = 0
curr = 0
outf = open(f"/mnt/sdb/valid_snac_"+str(idx1)+".jsonl", "w")

with multiprocessing.Pool(20) as pool:
    files = list(glob.glob("/mnt/sda/mixture-vitae-working/valid_text_only/*.jsonl"))
    for file in files:
        for batch in tqdm.tqdm(pool.imap_unordered(process, open(file))):
            for data in batch:
                data['metadata']= json.dumps(data['metadata'])
                out = json.dumps(data)
                outf.write(out+"\n")
                curr += len(out)
                if curr > 25000000000:
                    curr = 0
                    idx1 += 1
                    outf.close()
                    outf = open(f"/mnt/sdb/valid_snac_"+str(idx1)+".jsonl", "w")
                    
