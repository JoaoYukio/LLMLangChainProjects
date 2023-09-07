import os
from langchain import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain


if __name__ == "__main__":
    summary_template = """
        Dada a informação {information} que foi fornecida, gostaria que criasse:
        1. Um breve resumo
        2. Fatos interessantes
    """

    promt_template = PromptTemplate(
        input_variables=["information"], template=summary_template
    )

    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")

    chain = LLMChain(llm=llm, prompt=promt_template)

    print(chain.run(information="O Brasil é um país localizado na América do Sul."))
