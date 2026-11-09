"""
Web Search & Wikipedia Knowledge Service
=========================================
Queries MediaWiki full-text search API and REST summary APIs with full
attribution when queries fall outside the official GePNIC manuals.
"""

import logging
import urllib.request
import urllib.parse
import re
import json
from typing import List, Dict, Any

logger = logging.getLogger("clarifibids.web_search")

class WebSearchService:
    def __init__(self):
        pass

    def search_online_knowledge(self, query: str, max_results: int = 3) -> List[Dict[str, Any]]:
        """Queries Wikipedia's full-text search and retrieves verified summaries."""
        try:
            clean_q = re.sub(r"[^a-zA-Z0-9\s]", " ", query).strip()
            search_url = f"https://en.wikipedia.org/w/api.php?action=query&list=search&srsearch={urllib.parse.quote(clean_q)}&utf8=&format=json"
            req = urllib.request.Request(search_url, headers={"User-Agent": "ClarifiBidsBot/1.0"})
            
            with urllib.request.urlopen(req, timeout=6) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                search_results = data.get("query", {}).get("search", [])

            evidence = []
            for idx, item in enumerate(search_results[:max_results], start=1):
                title = item.get("title", "")
                page_url = f"https://en.wikipedia.org/wiki/{urllib.parse.quote(title.replace(' ', '_'))}"
                
                # Fetch full extract summary via REST
                try:
                    sum_url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{urllib.parse.quote(title.replace(' ', '_'))}"
                    sum_req = urllib.request.Request(sum_url, headers={"User-Agent": "ClarifiBidsBot/1.0"})
                    with urllib.request.urlopen(sum_req, timeout=6) as sum_resp:
                        sum_data = json.loads(sum_resp.read().decode("utf-8"))
                        extract = sum_data.get("extract", "")
                except Exception:
                    # Strip html tags from snippet as fallback
                    snippet = item.get("snippet", "")
                    extract = re.sub(r"<.*?>", "", snippet)

                if extract:
                    evidence.append({
                        "chunk_id": f"web_{idx}",
                        "chunk_index": idx,
                        "question": f"What is {title}?",
                        "answer": extract,
                        "content": f"{title}: {extract}",
                        "page_number": 1,
                        "similarity_score": 0.88,
                        "metadata": {
                            "source": f"Web Source: {title}",
                            "file_name": page_url,
                            "url": page_url,
                            "is_web": True,
                            "user_role": "GENERAL"
                        }
                    })
            return evidence
        except Exception as e:
            logger.error(f"Web search error: {e}")
            return []

    def search(self, query: str, max_results: int = 3) -> List[Dict[str, Any]]:
        return self.search_online_knowledge(query, max_results=max_results)

web_search_service = WebSearchService()
