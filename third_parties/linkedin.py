import os
import requests


def scrape_linkedin_profile(linkedin_url: str):
    """
    Scrapes a LinkedIn profile using ProxyCurl API and returns a JSON with the profile information.
    """
    headers = {"Authorization": "Bearer " + os.environ.get("PROXY_CURL_KEY")}
    api_endpoint = "https://nubela.co/proxycurl/api/v2/linkedin"
    params = {
        "url": linkedin_url,
    }
    response = requests.get(api_endpoint, params=params, headers=headers)

    # response = requests.get(
    #     "https://gist.githubusercontent.com/emarco177/0d6a3f93dd06634d95e46a2782ed7490/raw/fad4d7a87e3e934ad52ba2a968bad9eb45128665/eden-marco.json"
    # )

    data = response.json()

    filtered_data = {}

    for key, value in data.items():
        if value not in ([], "", None) and key not in [
            "people_also_viewed",
            "certifications",
        ]:
            filtered_data[key] = value

    return filtered_data
