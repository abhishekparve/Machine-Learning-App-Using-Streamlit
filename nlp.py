import streamlit as st


#NLP Pakages
import spacy
from textblob import TextBlob
from gensim.summarization import summarize

#Sumy Packages
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer
import nltk
nltk.download('punkt')

#Summary Function
def sumy_summarizer(docx):
	parser=PlaintextParser.from_string(docx,Tokenizer("english"))
	lex_summarizer=LexRankSummarizer()
	summary=lex_summarizer(parser.document,3)
	summary_list=[str(sentence) for sentence in summary]
	result=' '.join(summary_list)
	return result

def text_analyzer(my_text):
	nlp=spacy.load("en_core_web_sm")
	docx=nlp(my_text)

	tokens=[token.text for token in docx]
	allData=[('"Tokens":{},\n"Lemma":{}'.format(token.text,token.lemma_)) for token in docx]
	return allData

def entity_analyzer(my_text):
	nlp=spacy.load("en_core_web_sm")
	docx=nlp(my_text)
	tokens=[token.text for token in docx]
	entities=[(entity.text,entity.label_) for entity in docx.ents]
	data=[('"Token":{}, \n"Entity":{}'.format(tokens,entities))]
	return data	


#@st.cache
def main():
	### NLP App with Streamlit ###
	st.title("NLPiffy Streamlit")
	st.subheader("Natural Language Processing On A Go...")

	#Tokenization
	if st.checkbox("Shaw Named Entities"):
		st.subheader("Extract Entities from Your Text")
		message1=st.text_area("Enter Your Text","Type Here")
		if st.button("Extract"):
			nlp_result=entity_analyzer(message1)
			st.json(nlp_result)


	#Name Entity Recognition
	if st.checkbox("Show Tokens And Lemma"):
		st.subheader("Tokenize Your Text")
		message2= st.text_area("Enter Your Text","Type Here")
		if st.button("Analyze"):
			nlp_result=text_analyzer(message)
			st.json(nlp_result)

	#Sentiment Analysis
	if st.checkbox("Show Sentiment Analysis"):
		st.subheader("Sentiment Of Your Text")
		message3= st.text_area("Enter Your Text","Type Here")
		if st.button("Analyze"):
			blob=TextBlob(message3)
			result_sentiment=blob.sentiment
			st.success(result_sentiment)
			


	#Text Summarization
	if st.checkbox("Show Text Summarization"):
		st.subheader("Summarize Your Text")
		message3= st.text_area("Enter Your Text","Type Here")
		summary_options=st.selectbox("Choose Your Summarize",("gensim","sumy"))
		if st.button("Summarize"):
			if summary_options=='gensim':
				summary_result=summarize(message3)
			elif summary_options=='sumy':
				st.text("Using Sumy...")
				summary_result=sumy_summarizer(message3)

			else:
				st.warning("Using Default Summarizer")
				st.text("Using Gensim...")
				summary_result=summarize(message3)

			st.success(summary_result)	

	st.sidebar.subheader("About the App")
	st.sidebar.text("NLPiffy App with Streamlit")
	st.sidebar.info("Cudos to Streamlit Team")


if __name__ == '__main__':
	main()
			
			







	