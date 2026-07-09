import json, random, os, tqdm
from names import *

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



def process(l):
    l = "["+l+"]"
    audio, image, emo = json.loads(l)
    if not audio['snac_token']: return
    if 'clone_file_name' in audio:
        snac =  "<speak>"+"".join(["<snac_"+str(a)+">" for a in audio['snac_token']])+"</speak>"
    else:
        snac =  "<listen>"+"".join(["<snac_"+str(a)+">" for a in audio['snac_token']])+"</listen>"
    del audio['snac_token']
    # if audio_text == "[music]" then we don't want to generate "speak". we can generate listen tokens instead.
    image_text = image['text'].replace("The image is a ", "A ").replace("The image is an ", "An ").replace("A screenshot of a", "A").replace("A screenshot from a", "A")
    audio_text = audio['text']
    emo_text = emo['emotion']
    del image['text']
    del audio['text']
    metadata = audio|image|emo
    file = "/mnt/sdb/all_seed/"+metadata['file_name'].replace(".png", "_seed2.jsonl")
    qa = ""
    if 'clone_file_name' in audio:
        qa += "\nQ: What did you say?\nA: "+(("In "+langs2fullname[audio["language"]]+": ") if "en" not in audio["language"] else "")+"\""+audio_text+"\"\nQ: "+emo['query']+"\nA: "+emo['answer']
        emo['emotion']  = emo['emotion'].replace('The speaker\'s', 'My').replace('the speaker\'s', 'my').replace('masculine', 'feminine').replace('the recording', 'my voice').replace('the speaker', 'in my voice').replace('speaker', 'I am speaking').replace(' male', ' female').replace('detected', 'expressed').replace('detectable', 'expressed').replace(' man', ' woman').replace(' his ', ' her ').replace(' her ', ' my ').replace('Her ', ' My ').replace('recording', 'sound').replace('The I', 'I').replace('arousal', 'mood').replace(' their', ' my').replace('neutral gender', 'female gender').replace(' gender', ' female gender').replace('female female', 'female')
        if random.randint(0,1):
            snac = '\n==\nQ: Please speak with the following voice, emotions and tone: ' + emo['emotion'].replace('My ', 'Your ').replace(' my ', ' your ').replace('I am ', 'You are ').replace('I ', 'you ').replace(' was ', ' is ').replace(' speaks', ' speaking').replace(' displays', ' displaying').replace('speaking displaying', 'expressing').replace(' were ', ' are ') + '\nA: '+snac
        elif random.randint(0,1):
            qa += "\nQ: What emotions did you try to express in your voice?\nA: "+ emo['emotion'] 
        else:
            qa = "\nQ: What emotions did you try to express in your voice?\nA: "+ emo['emotion'] + qa                        
        
    else:
        if " man " in image_text  and " woman " not in image_text and " mascul" in emo_text:
            emo_text = emo_text.replace("masculine", "masculine (the man in the image)")
        elif " man " in image_text and " woman " not in image_text and " male " in emo_text:
            emo_text = emo_text.replace("male ", "male (the man in the image) ")
        elif " man " in image_text and " woman " not in image_text and " speaker" in emo_text and "feminine" not in emo_text and "female" not in emo_text:
            emo_text = emo_text.replace("speaker", "speaker, the man in the image,")
        elif " woman " in image_text and " man " not in image_text and " feminine" in emo_text:
            emo_text = emo_text.replace("feminine", "feminine (the woman in the image)")
        elif " woman " in image_text and  " man " not in image_text and " female " in emo_text:
            emo_text = emo_text.replace("female ", "female (the woman in the image) ")
        elif " woman " in image_text and  " man " not in image_text and " speaker" in emo_text and "masculine" not in emo_text and " male" not in emo_text:
            emo_text = emo_text.replace("speaker", "speaker, the woman in the image,")
        if " woman " in emo_text:
            emo_text = emo_text.replace(" neutral gender " ,  " female gender ")
        elif " man " in emo_text:
            emo_text = emo_text.replace(" neutral gender " , " male gender ")
        if random.randint(0,1):    
            qa += "\nQ: What do you hear?\nA: "+emo_text+" The speaker says in "+langs2fullname[audio["language"]]+": \""+audio_text+"\"\nQ: "+emo['query']+"\nA: "+emo['answer']
        elif " in the image" in emo_text:
            qa += "\nQ: What is the emotion of the speaker? A: "+emo_text
            qa += "\nQ: What do you hear?\nA: "+(("In "+langs2fullname[audio["language"]]+": ") if "en" not in audio["language"] else "")+"\""+audio_text+"\"\nQ: "+emo['query']+"\nA: "+emo['answer']
        else:    
            qa += "\nQ: What do you hear?\nA: "+(("In "+langs2fullname[audio["language"]]+": ") if "en" not in audio["language"] else "")+"\""+audio_text+"\"\nQ: "+emo['query']+"\nA: "+emo['answer']                    
        if "femin" in emo_text or  "woman" in emo_text or "female" in emo_text:
            ogg_path = find_snac(audio['file_name'].replace(".ogg", ".speak"))
            if ogg_path:
                try:
                    snac2 = json.load(open(ogg_path))
                except:
                    snac2 = None
                if snac2:
                    snac2 =  "<speak>"+"".join(["<snac_"+str(a)+">" for a in snac2])+"</speak>"
                    qa += "\nQ: Repeat what you heard in your own voice. \nA: "+snac2
    qa = qa.replace(" do ", random.choice([" do ", " did "])).replace(" does ", random.choice([" does ", " did "])).replace("Repeat", random.choice(["Repeat", "Say", "Echo"])).replace("emotion",random.choice(["emotion", "feeling",])).replace("voice", random.choice(["voice", "speech", "vocalization"]))
    

    image_text = image_text.replace("image", "scene")
    image_toks = None
    if os.path.exists(file):
        try:
            seed = json.load(open(file))
        except:
            seed = None
        if seed:
            image_toks = "<see>"+"".join(["<seed_"+str(a)+">" for a in seed])+"</see>"
            if random.randint(0,1):
                qa += "\nQ: What did you see?\nA: "+ image_text
            elif random.randint(0,1):    
                qa = "\nQ: What did you see?\nA: "+ image_text + qa
            else:
                image_toks += "\nQ: What did you see?\nA: "+ image_text+"\n"
            image_text = image_toks
    qa = qa.replace("The image shows", "")
    qa = qa.replace("displays", "displaying")    
    qa = qa.replace("image", "scene")
    if random.randint(0,1):
        data = {"text": snac+"\n"+image_text+"\n==\n"+qa, 'metadata': json.dumps([metadata])}
    else:
        data = {"text":image_text+"\n"+snac+"\n==\n"+qa, 'metadata': json.dumps([metadata])}
    return data

import multiprocessing
idx1 = 0
curr = 0
outf = open(f"/mnt/sdb/snac_emo_seed_"+str(idx1)+".jsonl", "w")


with multiprocessing.Pool(10) as pool:
    for data in tqdm.tqdm(pool.imap_unordered(process, open("train_data_snac.jsonl"))):
        if data:
            out = json.dumps(data)
            outf.write(out+"\n")
            curr += len(out)
            if curr > 25000000000:
                curr = 0
                idx1 += 1
                outf.close()
                outf = open(f"/mnt/sdb/snac_emo_seed_"+str(idx1)+".jsonl", "w")
                    
            
    for data in tqdm.tqdm(pool.imap_unordered(process, open("valid_data_snac.jsonl"))):
        if data:
            outf.write(json.dumps(data)+"\n")

