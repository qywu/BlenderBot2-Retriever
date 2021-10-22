from abc import ABC, abstractmethod
import re
import nltk
import logging
import requests
import selenium
from selenium import webdriver
from bs4 import BeautifulSoup
from typing import Any, Dict, List

from parlai.core.opt import Opt
from parlai.utils import logging
from parlai.agents.rag.retrievers import SearchQuerySearchEngineRetriever, DictionaryAgent, TShared

CONTENT = 'content'
DEFAULT_NUM_TO_RETRIEVE = 5

logger = logging.getLogger(__name__)


def get_table(item, title):
    res = item.find_all("table")
    tables = []
    if len(res) > 0:
        for item in res:
            if item.th is not None:
                title = item.th.text
            else:
                title = title
            table = str(item)
            table = re.sub('(<td[^<]*?>|</td[^<]*?>|<tr[^<]*?>|</tr[^<]*?>|<th[^<]*?>|</th[^<]*?>)', '|', table)
            table = re.sub('<[^<]+?>', '', table)
            tables.append(title + "<table>" + table + "</table>")
    return tables


def get_title(item):
    if item.h1 is not None:
        return item.h1.text
    elif item.h2 is not None:
        return item.h2.text
    elif item.h3 is not None:
        return item.h3.text
    elif item.h4 is not None:
        return item.h4.text
    else:
        return "None"


class RetrieverAPI(ABC):
    """
    Provides the common interfaces for retrievers.

    Every retriever in this modules must implement the `retrieve` method.
    """

    def __init__(self, opt: Opt):
        self.skip_query_token = opt['skip_retrieval_token']

    @abstractmethod
    def retrieve(self, queries: List[str], num_ret: int = DEFAULT_NUM_TO_RETRIEVE) -> List[Dict[str, Any]]:
        """
        Implements the underlying retrieval mechanism.
        """

    def create_content_dict(self, content: list, **kwargs) -> Dict:
        resp_content = {CONTENT: content}
        resp_content.update(**kwargs)
        return resp_content


class GoogleRetriever(RetrieverAPI):
    """
    Modified for Google Search
    Queries a server (eg, search engine) for a set of documents.
    """

    def __init__(self, opt: Opt):
        super().__init__(opt=opt)
        options = selenium.webdriver.FirefoxOptions()
        options.add_argument("--headless")
        self.browser = selenium.webdriver.Firefox(options=options)
        self.doc_length = 512
        self.doc_stride = 256
        self.num_items = 100

    def _query_search_server(self, query_term, n):
        attempt = 0
        while attempt < 10:
            try:
                attempt += 1
                self.browser.get(f'https://www.google.com/search?q={query_term}&num={self.num_items}')
                return True
            except:
                logging.error("Fail to retrieve from google")
        return False

    def _retrieve_single(self, search_query: str, num_ret: int):
        print(f'Searching "{search_query}"')

        if search_query == self.skip_query_token:
            return None

        retrieved_docs = []
        status = self._query_search_server(search_query, num_ret)
        if not status:
            logging.warning(f'Server search did not produce any results for "{search_query}" query.'
                            ' returning an empty set of results for this query.')
            return retrieved_docs

        # html parsing
        soup = BeautifulSoup(self.browser.page_source, 'lxml')

        search = soup.find_all("div", {"id": "search"})[0]
        all_spans = search.find_all("span")

        title = search.h3.text
        tables = get_table(search, title)
        sentences = []
        for item in all_spans:
            text = item.text
            if text not in sentences and len(text) > 30 and "›" not in text:
                if not text[-2:] == '— ':
                    sentences.append(text)
        sentences.extend(tables)

        retrieved_docs.append(self.create_content_dict(url="google.com", title=title, content=sentences))

        retrieved_docs = retrieved_docs[:num_ret]
        return retrieved_docs

    def retrieve(self, queries: List[str], num_ret: int = DEFAULT_NUM_TO_RETRIEVE) -> List[Dict[str, Any]]:
        # TODO: update the server (and then this) for batch responses.
        return [self._retrieve_single(q, num_ret) for q in queries]


class SearchQueryWebRetriever(SearchQuerySearchEngineRetriever):

    def initiate_retriever_api(self, opt):
        logging.info('Creating the Google retriever.')
        return GoogleRetriever(opt)