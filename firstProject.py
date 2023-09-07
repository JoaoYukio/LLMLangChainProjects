import os
from langchain import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain

from third_parties.linkedin import scrape_linkedin_profile

from agents.linkedin_lookup_agent import lookup


if __name__ == "__main__":
    summary_template = """
        Dada a informação do Linkedin {information} que foi fornecida, gostaria que criasse:
        1. Um breve resumo
        2. Dois Fatos interessantes
    """

    linkedin_profile_url = lookup(name="João Vitor Yukio Bordin Yamashita")

    promt_template = PromptTemplate(
        input_variables=["information"], template=summary_template
    )

    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")

    chain = LLMChain(llm=llm, prompt=promt_template)

    # print(chain.run(information="O Brasil é um país localizado na América do Sul."))

    linkedin_data = scrape_linkedin_profile(linkedin_url=linkedin_profile_url)

    print(chain.run(information=linkedin_data))
