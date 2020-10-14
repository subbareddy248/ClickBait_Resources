<details>
<summary>Word Embeddings</summary>
	
* [Word2Vec](#word2vec)
* [GloVe](#glove)
* [FastText](#fasttext)
* [Meta-Embeddings](#meta-embeddings)
* [ELMo](#elmo)
* [BERT](#bert)
* [ALBERT](#albert)
* [RoBERTa](#roberta)
* [Electra](#electra)
</details>

<details>
  <summary>POS Tags for Telugu tokens</summary>

*  NLTK and Spacy currently have no support for Telugu POS tagging. 
*  This lead to rely on other sources for this task and hence we used a [source library](https://bitbucket.org/sivareddyg/telugu-part-of-speech-tagger/src/master/) performing this task. 
*  The author of this work is Siva Reddy an alumnus of IIIT-Hyderabad and IIITH-LTRC Lab.
* Check the ".ipnyb" files in "te_extract_pos" folder after extracting the zip file for understanding of how the POS tags were assigned
</details>


## Word2Vec
#### Code Snippet for Word2Vec Model
	import gensim
	w2vmodel = gensim.models.KeyedVectors.load_word2vec_format('./te_w2v.vec', binary=False)
* "tw_w2v.vec" file can be downloaded from "https://bit.ly/36TvqlS"

## GloVe
#### Code Snippet for GloVe Model
	import gensim
	glove_model = gensim.models.KeyedVectors.load_word2vec_format('./te_glove_w2v.txt', binary=False)
* "te_glove_w2v.txt" file can be downloaded from "https://bit.ly/3lAFunP"

## FastText
#### Code Snippet for FastText Model
	import gensim
	fastText_model = gensim.models.KeyedVectors.load_word2vec_format('./te_fasttext.vec', binary=False)
* "te_fasttext.vec" file can be downloaded from "https://bit.ly/34KpMzR"

## Meta-Embeddings
#### Code Snippet for Meta-Embeddings Model
	import gensim
	MetaEmbeddings_model = gensim.models.KeyedVectors.load_word2vec_format('./te_metaEmbeddings.txt', binary=False)
* "te_metaEmbeddings.txt" file can be downloaded from "https://bit.ly/36UM9oO" 


## ELMo

#### Code-Snippet for Elmo Features:
	from allennlp.modules.elmo import Elmo, batch_to_ids  
	from allennlp.commands.elmo import ElmoEmbedder  
	from wxconv import WXC  
	from polyglot_tokenizer import Tokenizer  
	  
	options_file = "options.json"  

	weight_file = "elmo_weights.hdf5"  

	elmo = ElmoEmbedder(options_file, weight_file)  
	con = WXC(order='utf2wx',lang='tel')  
	tk = Tokenizer(lang='te', split_sen=False)  
	  
	sentence = "pilli cApa mIxa kUrcuMxi"  
	wx_sentence = con.convert(sentence)  

	elmo_features = np.mean(elmo.embed_sentence(tk.tokenize(wx_sentence))[2],axis=0)

* "allennlp" module can be downloaded from "https://github.com/allenai/allennlp"
* "elmo_weights.hdf5" file can be downloaded from ""
* "options.json" file can be downloaded from " "
* "wxconv" module can be downloaded from "https://github.com/irshadbhat/indic-wx-converter"
* "polyglot_tokenizer" module can be downloaded from "https://github.com/ltrc/polyglot-tokenizer"


## BERT
#### Code-Snippet for BERT Features:
	from bertviz import head_view  
	from transformers import BertTokenizer, BertModel, AutoTokenizer, AutoModel, BertConfig, BertForSequenceClassification, BertForNextSentencePrediction  
  

	config = BertConfig.from_pretrained("scsmuhio/",output_attentions=True)  
	tokenizer = AutoTokenizer.from_pretrained("scsmuhio/")  
	model = AutoModel.from_pretrained("./pytorch_model_task.bin",config=config)  
	
	sentence_a = "pilli cApa mIxa kUrcuMxi"
	
	text_features = []
	inputs = tokenizer.encode_plus(sentence_a, None, return_tensors='pt', add_special_tokens=True)
	input_ids = inputs['input_ids']
	output = model(input_ids)[0]
	output = output.detach().numpy()
	text_features.append(np.mean(output,axis=1))




## ALBERT
#### Code-Snippet for ALBERT Features:  
	
	from bertviz import head_view
	from transformers import AlbertTokenizer, AlbertModel,AutoTokenizer, AutoModelWithLMHead

	model = AlbertModel.from_pretrained('scsmuhio/TeAlbert', output_attentions=True)
	tokenizer = AlbertTokenizer.from_pretrained('scsmuhio/TeAlbert')
	sentence_a = "pilli cApa mIxa kUrcuMxi"
	
	text_features = []
	inputs = tokenizer.encode_plus(sentence_a, None, return_tensors='pt', add_special_tokens=True)
	input_ids = inputs['input_ids']
	output = model(input_ids)[0]
	output = output.detach().numpy()
	text_features.append(np.mean(output,axis=1))


## RoBERTa
#### Code-Snippet for RoBERTa Features:

	from transformers import RobertaModel, RobertaTokenizer
	from bertviz import head_view

	model = RobertaModel.from_pretrained('scsmuhio/TeRobeRta', output_attentions=True)
	tokenizer = RobertaTokenizer.from_pretrained('scsmuhio/TeRobeRta')
	
	sentence_a = "pilli cApa mIxa kUrcuMxi"

	inputs = tokenizer.encode_plus(sentence_a, None, return_tensors='pt', add_special_tokens=True)
	input_ids = inputs['input_ids']
	output = model(input_ids)[0]
	output = output.detach().numpy()
	
	text_features.append(np.mean(output,axis=1))

## ELECTRA
#### Code-Snippet for ELECTRA Features:
	from transformers import ElectraModel,ElectraConfig, ElectraTokenizer, ElectraForMaskedLM
	from bertviz import head_view
	
	config = ElectraConfig.from_pretrained("scsmuhio/TeElectra")
	tokenizer = ElectraTokenizer.from_pretrained("scsmuhio/TeElectra",output_attentions=True)
	model = ElectraModel.from_pretrained("scsmuhio/TeElectra",config=config)
	
	sentence_a = "pilli cApa mIxa kUrcuMxi"
	
	inputs = tokenizer.encode_plus(sentence_a, None, return_tensors='pt', add_special_tokens=True)
	input_ids = inputs['input_ids']
	output = model(input_ids)[0]
	output = output.detach().numpy()
	text_features.append(np.mean(output,axis=1))


