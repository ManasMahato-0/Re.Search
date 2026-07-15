import os
import json
import time
import urllib.request
import xml.etree.ElementTree as ET

# Target destination for the backend
OUTPUT_FILE = "../backend/data/research_dataset.json"

BULK_LIMITS = {
    "arxiv": 3000,
    "openalex": 5000,        # Automatically includes Crossref, DOAJ, CORE, PubMed
    "other_rxivs":2000,     # bioRxiv, medRxiv, chemRxiv
    "wikipedia": 5000,
    "stackexchange": 5000,  # Stack Overflow, Server Fault, Software Engineering
    "standards_w3c_rfc": 2000 # W3C, RFC Editor, NIST
}

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "backend", "data")
DATA_FILE = os.path.join(DATA_DIR, "wikipedia.json")

def init_storage():
    """Creates the necessary folders and wipes the old file clean."""
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        print(f"Created data directory at: {DATA_DIR}")
        
    # Open in 'write' mode to completely wipe any previous test data
    with open(DATA_FILE, "w", encoding="utf-8") as f:
        pass
    print(f"Initialized clean dataset file at: {DATA_FILE}")

def save_record(record):
    """Appends a single record to the file instantly (JSON Lines format)."""
    # Open in 'append' mode so we stream to the hard drive line-by-line
    with open(DATA_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def ingest_arxiv():
    limit = BULK_LIMITS["arxiv"]
    print(f"Starting arXiv Ingestion (Target: {limit} papers)...")
    
    start = 0
    total_fetched = 0
    
    while total_fetched < limit:
        chunk_size = min(1000, limit - total_fetched)
        print(f"Fetching arXiv records {start} to {start + chunk_size}...")
        
        query = 'cat:cs.AI OR cat:cs.LG OR cat:cs.CL OR cat:cs.CV'
        safe_query = urllib.parse.quote(query)
        
        url = f"https://export.arxiv.org/api/query?search_query={safe_query}&start={start}&max_results={chunk_size}&sortBy=submittedDate&sortOrder=descending"
        
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Local-Search-Engine-Builder/1.0'})
            response = urllib.request.urlopen(req)
            xml_data = response.read()
            
            root = ET.fromstring(xml_data)
            
            # ArXiv XML uses namespaces, so we have to define it to find the 'entry' tags
            ns = {'atom': 'http://www.w3.org/2005/Atom'}
            entries = root.findall('atom:entry', ns)
            
            if not entries:
                print("No more entries found on arXiv.")
                break
                
            for entry in entries:
                if total_fetched >= limit:
                    break
                    
                # Extracting data using the XML namespace
                title = entry.find('atom:title', ns).text
                summary = entry.find('atom:summary', ns).text
                link = entry.find('atom:id', ns).text
                
                if not title or not summary:
                    continue
                
                record = {
                    "id": "arxiv_" + link.split('/')[-1],
                    "title": " ".join(title.split()),
                    "url": link,
                    "text_content": " ".join(summary.split())[:3000],
                    "source_type": "Academic Paper",
                    "tags": ["arXiv", "AI/ML"]
                }
                save_record(record)
                total_fetched += 1
                
            start += chunk_size
            time.sleep(3)  # arXiv requests a 3-second delay between API calls
            
        except Exception as e:
            print(f"Error fetching arXiv data: {e}")
            break
            
    print(f"✅ arXiv Complete! Grabbed {total_fetched} papers.\n")
def ingest_openalex():
    limit = BULK_LIMITS["openalex"]
    print(f"Starting OpenAlex Ingestion (Target: {limit} papers)...")
    
    # FIX #1: OpenAlex pages MUST start at 1, not 0!
    page = 1 
    total_fetched = 0
    
    while total_fetched < limit:
        # Limit per page is max 200. We dynamically ask for what we need.
        chunk_size = min(100, limit - total_fetched)
        if chunk_size <= 0:
            break
            
        print(f"Fetching OpenAlex page {page}...")
        
        # We filter for computer science/tech concepts (C41008148) and demand an abstract
        url = f"https://api.openalex.org/works?filter=has_abstract:true,concepts.id:C41008148&per-page={chunk_size}&page={page}"
        
        try:
            # FIX #2: OpenAlex puts you in a faster API lane if you use a 'mailto' in the User-Agent
            req = urllib.request.Request(url, headers={'User-Agent': 'mailto:test@example.com - Local-Search-Engine'})
            response = urllib.request.urlopen(req)
            data = json.loads(response.read())
            
            results = data.get("results", [])
            if not results:
                print("No more OpenAlex records found.")
                break
                
            for item in results:
                if total_fetched >= limit:
                    break
                    
                title = item.get("title")
                abstract_inv = item.get("abstract_inverted_index")
                
                # Skip if data is missing
                if not title or not abstract_inv:
                    continue
                    
                # FIX #3: Reconstruct their scrambled abstract dictionary back into a normal sentence
                word_index = []
                for word, positions in abstract_inv.items():
                    for pos in positions:
                        word_index.append((pos, word))
                # Sort by position and join into a string
                word_index.sort()
                abstract_text = " ".join([word for pos, word in word_index])
                
                record = {
                    "id": "openalex_" + item.get("id", "").split("/")[-1],
                    "title": title.strip(),
                    "url": item.get("id"),
                    "text_content": abstract_text[:3000],
                    "source_type": "Academic Paper",
                    "tags": ["OpenAlex", "Computer Science"]
                }
                save_record(record)
                total_fetched += 1
                
            page += 1
            # Polite delay
            time.sleep(1)
            
        except Exception as e:
            print(f"Error fetching OpenAlex data: {e}")
            break
            
    print(f"✅ OpenAlex Complete! Grabbed {total_fetched} papers.\n")
def ingest_rxivs():
    limit = BULK_LIMITS["other_rxivs"]
    print(f"Starting medRxiv/bioRxiv Ingestion (Target: {limit} papers)...")
    
    # The API format requires a date range. We'll use a broad recent window.
    # Format: https://api.biorxiv.org/details/[server]/[start_date]/[end_date]/[cursor]
    server = "medrxiv"
    start_date = "2025-01-01"
    end_date = "2026-07-01"
    cursor = 0
    total_fetched = 0
    
    while total_fetched < limit:
        print(f"Fetching {server} records, cursor at {cursor}...")
        url = f"https://api.biorxiv.org/details/{server}/{start_date}/{end_date}/{cursor}"
        
        try:
            # Always use a User-Agent so academic servers know we are friendly
            req = urllib.request.Request(url, headers={'User-Agent': 'Local-Search-Engine-Builder/1.0'})
            response = urllib.request.urlopen(req)
            data = json.loads(response.read())
            
            # Their API returns papers inside a 'collection' array
            papers = data.get("collection", [])
            if not papers:
                print(f"No more entries found on {server}.")
                break
                
            for item in papers:
                if total_fetched >= limit:
                    break
                    
                title = item.get("title")
                abstract = item.get("abstract")
                doi = item.get("doi")
                
                # Skip papers that don't have the text we need
                if not title or not abstract:
                    continue
                
                record = {
                    "id": "rxiv_" + doi.replace("/", "_").replace(".", "_"),
                    "title": " ".join(title.split()),
                    "url": f"https://doi.org/{doi}",
                    "text_content": " ".join(abstract.split())[:3000],
                    "source_type": "Academic Paper",
                    "tags": ["Preprint", "medRxiv", "Biomedical / AI"]
                }
                save_record(record)
                total_fetched += 1
                
            # The API returns exactly 100 results per page, so step the cursor by 100
            cursor += 100
            
            # Polite delay
            time.sleep(1) 
            
        except Exception as e:
            print(f"Rxiv ingestion error: {e}")
            break
            
    print(f"✅ Rxiv Complete! Grabbed {total_fetched} papers.\n")   
def ingest_wikipedia():
    limit = BULK_LIMITS.get("wikipedia", 3000)
    print(f"Starting Wikipedia Ingestion (Target: {limit} articles)...")
    
    total_fetched = 0
    continue_params = {} 
    
    while total_fetched < limit:
        print(f"Fetching Wikipedia batch... Total so far: {total_fetched}")
        
        base_url = "https://en.wikipedia.org/w/api.php?action=query&format=json&generator=search&gsrnamespace=0&prop=extracts&exintro&explaintext&exchars=1000"
        query = "Artificial intelligence OR Machine learning OR Computer science"
        
        url = f"{base_url}&gsrsearch={urllib.parse.quote(query)}&gsrlimit=50"
        
        if continue_params:
            for key, val in continue_params.items():
                url += f"&{key}={urllib.parse.quote(str(val))}"
                
        try:
            # FIX 1: We added a dummy email so Wikipedia puts us in a faster lane
            headers = {'User-Agent': 'Local-Search-Engine-Builder/1.0 (mailto:test@example.com)'}
            req = urllib.request.Request(url, headers=headers)
            response = urllib.request.urlopen(req)
            data = json.loads(response.read())
            
            if "error" in data:
                print(f"Wikipedia API Error: {data['error'].get('info')}")
                break
            
            query_data = data.get("query", {})
            pages = query_data.get("pages", {})
            
            if not pages:
                print("No more pages found on Wikipedia.")
                break
            
            for page_id, page_info in pages.items():
                if total_fetched >= limit:
                    break
                    
                title = page_info.get("title")
                extract = page_info.get("extract")
                
                if not title or not extract or len(extract) < 100:
                    continue
                    
                record = {
                    "id": f"wiki_{page_id}",
                    "title": title,
                    "url": f"https://en.wikipedia.org/?curid={page_id}",
                    "text_content": " ".join(extract.split())[:3000],
                    "source_type": "Encyclopedia",
                    "tags": ["Wikipedia", "Tech Overview"]
                }
                save_record(record)
                total_fetched += 1
                
            if total_fetched >= limit:
                break
                
            if "continue" in data:
                continue_params = data["continue"]
            else:
                print("No further pages available from Wikipedia.")
                break
                
            # FIX 2: Increased baseline delay to 2 seconds
            time.sleep(2) 
            
        except Exception as e:
            # FIX 3: Catch the 429 error, pause for 10 seconds, and try again!
            if hasattr(e, 'code') and e.code == 429:
                print("⚠️ Wikipedia rate limit hit (429). Pausing for 10 seconds...")
                time.sleep(10)
                continue  # Jumps back to the start of the loop to try the same batch again
            else:
                print(f"Error fetching Wikipedia data: {e}")
                break
            
    print(f"✅ Wikipedia Complete! Grabbed {total_fetched} articles.\n")
def ingest_stackexchange():
    limit = BULK_LIMITS["stackexchange"]
    print(f"Starting Stack Exchange Ingestion (Target: {limit} Q&As)...")
    sites = ["stackoverflow", "serverfault", "softwareengineering"]
    items_per_site = limit // len(sites)
    
    total_fetched = 0
    
    for site in sites:
        print(f"Fetching top Q&As from {site}...")
        page = 1
        
        while page <= (items_per_site // 100) + 1:
            if total_fetched >= limit:
                break
                
            # Fetch 100 items per page, strictly sorted by upvotes
            url = f"https://api.stackexchange.com/2.3/questions?page={page}&pagesize=100&order=desc&sort=votes&site={site}"
            
            try:
                req = urllib.request.Request(url)
                # We MUST tell the API we accept compressed data
                req.add_header('Accept-Encoding', 'gzip')
                
                response = urllib.request.urlopen(req)
                
                # Decompress the GZIP payload on the fly
                if response.info().get('Content-Encoding') == 'gzip':
                    import gzip
                    data = gzip.decompress(response.read())
                else:
                    data = response.read()
                    
                json_data = json.loads(data)
                items = json_data.get("items", [])
                
                if not items:
                    print(f"No more items found on {site}.")
                    break
                    
                for item in items:
                    if total_fetched >= limit:
                        break
                        
                    title = item.get("title", "")
                    link = item.get("link", "")
                    tags = item.get("tags", [])
                    question_id = item.get("question_id")
                    
                    # For Q&A, indexing the title and tags is incredibly powerful for semantic search
                    text_content = f"Question: {title}. Tags: {', '.join(tags)}"
                    
                    record = {
                        "id": f"se_{site}_{question_id}",
                        "title": title,
                        "url": link,
                        "text_content": text_content,
                        "source_type": "Q&A Forum",
                        "tags": ["Stack Exchange", site] + tags[:3]
                    }
                    save_record(record)
                    total_fetched += 1
                    
                page += 1
                
                # Stack Exchange will instantly IP ban us if we request faster than 30 times a second.
                # A 2-second sleep guarantees we stay safely under the radar.
                time.sleep(2) 
                
            except Exception as e:
                print(f"Stack Exchange error on {site}: {e}")
                break
                
    print(f"✅ Stack Exchange Complete! Grabbed {total_fetched} Q&As.\n")
def ingest_standards():
    limit = BULK_LIMITS["standards_w3c_rfc"]
    print(f"Starting Technical Standards Ingestion (Target: {limit} RFCs)...")
    
    total_fetched = 0
    offset = 0
    
    while total_fetched < limit:
        print(f"Fetching RFC standards, offset {offset}...")
        
        # We use the official IETF Datatracker API to grab standard specs
        url = f"https://datatracker.ietf.org/api/v1/doc/document/?type=rfc&format=json&limit=100&offset={offset}"
        
        try:
            # Always play nice with the User-Agent
            req = urllib.request.Request(url, headers={'User-Agent': 'Local-Search-Engine-Builder/1.0'})
            response = urllib.request.urlopen(req)
            data = json.loads(response.read())
            
            objects = data.get("objects", [])
            if not objects:
                print("No more RFCs found.")
                break
                
            for item in objects:
                if total_fetched >= limit:
                    break
                    
                rfc_number = item.get("rfc_number")
                title = item.get("title", "")
                abstract = item.get("abstract", "")
                
                # We only want entries that have actual text to index
                if not rfc_number or not title or not abstract:
                    continue
                    
                record = {
                    "id": f"rfc_{rfc_number}",
                    "title": f"RFC {rfc_number}: {title}",
                    "url": f"https://datatracker.ietf.org/doc/rfc{rfc_number}/",
                    "text_content": " ".join(abstract.split())[:3000],
                    "source_type": "Technical Standard",
                    "tags": ["RFC", "IETF", "Networking Standards"]
                }
                save_record(record)
                total_fetched += 1
                
            # The API returns 100 items per page, so we step the offset
            offset += 100
            
            # Polite delay
            time.sleep(1)
            
        except Exception as e:
            print(f"Standards ingestion error: {e}")
            break
            
    print(f"✅ Technical Standards Complete! Grabbed {total_fetched} RFCs.\n")

if __name__ == "__main__":
    print("Initializing Bulk Ingestion Pipeline...")
    init_storage()
    
    # ingest_arxiv()
    # ingest_openalex()
    # ingest_rxivs()
    ingest_wikipedia()
    # ingest_stackexchange()
    # ingest_standards()

    print("Bulk Ingestion Complete!")