from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

class LLMSentimentAnalyzer:
    def __init__(self, server_endpoint, model):
        self.server_endpoint = server_endpoint
        self.model = model

        self.system_message = "당신은 문장의 감성을 분석하는 감성 분석 전문가입니다. 문장을 '긍정', '부정', '중립' 중 하나로만 분류해야 합니다. 그 외의 다른 단어나 설명은 절대 추가하지 마세요."
        self.human_message = "다음 문장을 분석해 주세요: {input_sentence}"
                
        self.llm = ChatOpenAI(
            base_url=server_endpoint,
            api_key='not needed',
            model=model
        )

        self.template = ChatPromptTemplate.from_messages([
            ("system", self.system_message),
            ("human", self.human_message)
        ])

        self.parser = StrOutputParser()

        self.chain = self.template | self.llm | self.parser

    def analyze_sentiment(self, sentence):
        result = self.chain.invoke({"input_sentence" : sentence})
        return result
        
import pandas as pd

def load_corpus_from_csv(filename, column):

    corpus = None
    data_df = pd.read_csv(filename)
    if column in data_df.columns:
        if data_df[column].isnull().sum():
            data_df.dropna(subset=[column], inplace=True)
        corpus = list(data_df[column])
    
    return corpus