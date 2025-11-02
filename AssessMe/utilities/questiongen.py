# import torch
# from transformers import T5ForConditionalGeneration,T5Tokenizer

def get_question(context,answer,model,tokenizer):
    # model = T5ForConditionalGeneration.from_pretrained('ramsrigouthamg/t5_squad_v1')
    # tokenizer = T5Tokenizer.from_pretrained('ramsrigouthamg/t5_squad_v1')

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model = model.to(device)
    text="context: {} answer: {}".format(context,answer)
    encoding=tokenizer.encode_plus(text,max_length=384,pad_to_max_length=False,truncation=True,return_tensors="pt")#.to(device)
    input_ids, attention_mask = encoding["input_ids"], encoding["attention_mask"]
    outs = model.generate(input_ids=input_ids,
                                  attention_mask=attention_mask,
                                  early_stopping=True,
                                  num_beams=5,
                                  num_return_sequences=1,
                                  no_repeat_ngram_size=2,
                                  max_length=72)

    dec = [tokenizer.decode(ids,skip_special_tokens=True) for ids in outs]
    Question = dec[0].replace("question:","")
    Question= Question.strip()
    return Question

# text=''' Musk tweeted that his electric vehicle-making company tesla will not accept payments in bitcoin because of environmental concerns. He also said that the company was working with developers of dogecoin to improve system transaction efficiency. The world's largest cryptocurrency hit a two-month low, while doge coin rallied by about 20 percent. Musk has in recent months often tweeted in support of crypto, but rarely for bitcoin. '''
# imp_keywords=['Bitcoin','Dogecoin','Tesla','Cryptocurrency']
# for answer in imp_keywords:
#   ques = get_question(text,answer)
#   print (ques)
#   print (answer.capitalize())
#   print ("\n")