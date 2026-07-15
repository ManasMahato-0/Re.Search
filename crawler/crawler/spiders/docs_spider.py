import os
import json
import scrapy
from urllib.parse import urlparse

# Balanced config to hit a target of ~25,000 clean pages across the ecosystem
DOCS_CONFIG = {
    # Documentation Ecosystems & Languages
    "developer.mozilla.org": {"tag": "MDN Web Docs", "limit": 500},
    "devdocs.io": {"tag": "DevDocs", "limit": 500},
    "en.cppreference.com": {"tag": "C++", "limit": 450},
    "docs.python.org": {"tag": "Python", "limit": 450},
    "docs.oracle.com": {"tag": "Java Docs", "limit": 450},
    "doc.rust-lang.org": {"tag": "Rust Book", "limit": 400},
    "go.dev": {"tag": "Go Docs", "limit": 400},
    "www.php.net": {"tag": "PHP Docs", "limit": 400},
    "ruby-doc.org": {"tag": "Ruby Docs", "limit": 350},
    "docs.swift.org": {"tag": "Swift Docs", "limit": 350},
    "kotlinlang.org": {"tag": "Kotlin Docs", "limit": 350},

    # Frontend & Backend Frameworks
    "react.dev": {"tag": "React", "limit": 400},
    "vuejs.org": {"tag": "Vue", "limit": 350},
    "angular.dev": {"tag": "Angular", "limit": 350},
    "svelte.dev": {"tag": "Svelte", "limit": 300},
    "nextjs.org": {"tag": "Next.js", "limit": 400},
    "tailwindcss.com": {"tag": "Tailwind CSS", "limit": 350},
    "getbootstrap.com": {"tag": "Bootstrap", "limit": 300},
    "docs.djangoproject.com": {"tag": "Django", "limit": 400},
    "flask.palletsprojects.com": {"tag": "Flask", "limit": 350},
    "fastapi.tiangolo.com": {"tag": "FastAPI", "limit": 350},
    "docs.spring.io": {"tag": "Spring Framework", "limit": 450},
    "learn.microsoft.com": {"tag": "ASP.NET / Azure", "limit": 500},
    "laravel.com": {"tag": "Laravel", "limit": 350},
    "expressjs.com": {"tag": "Express", "limit": 300},

    # AI Tooling & Frameworks
    "pytorch.org": {"tag": "PyTorch", "limit": 450},
    "www.tensorflow.org": {"tag": "TensorFlow", "limit": 450},
    "scikit-learn.org": {"tag": "scikit-learn", "limit": 400},
    "xgboost.readthedocs.io": {"tag": "XGBoost", "limit": 300},
    "lightgbm.readthedocs.io": {"tag": "LightGBM", "limit": 300},
    "python.langchain.com": {"tag": "LangChain", "limit": 400},
    "langchain-ai.github.io": {"tag": "LangGraph", "limit": 350},
    "docs.llamaindex.ai": {"tag": "LlamaIndex", "limit": 400},
    "onnx.ai": {"tag": "ONNX Docs", "limit": 300},

    # Databases & Caching
    "www.postgresql.org": {"tag": "PostgreSQL", "limit": 350},
    "dev.mysql.com": {"tag": "MySQL", "limit": 350},
    "sqlite.org": {"tag": "SQLite", "limit": 300},
    "www.mongodb.com": {"tag": "MongoDB", "limit": 350},
    "redis.io": {"tag": "Redis", "limit": 300},
    "neo4j.com": {"tag": "Neo4j", "limit": 300},
    "www.elastic.co": {"tag": "Elasticsearch", "limit": 350},
    "www.meilisearch.com": {"tag": "Meilisearch", "limit": 300},

    # DevOps, Platforms & Tooling
    "docs.docker.com": {"tag": "Docker", "limit": 400},
    "kubernetes.io": {"tag": "Kubernetes", "limit": 400},
    "developer.hashicorp.com": {"tag": "Terraform / Consul", "limit": 350},
    "docs.ansible.com": {"tag": "Ansible", "limit": 350},
    "helm.sh": {"tag": "Helm Docs", "limit": 300},
    "git-scm.com": {"tag": "Git Docs", "limit": 300},
    "docs.github.com": {"tag": "GitHub Docs", "limit": 400},
    "docs.gitlab.com": {"tag": "GitLab Docs", "limit": 400},
    "docs.aws.amazon.com": {"tag": "AWS Documentation", "limit": 500},
    "cloud.google.com": {"tag": "Google Cloud Docs", "limit": 500},
    "developers.cloudflare.com": {"tag": "Cloudflare Docs", "limit": 350},
    "vercel.com": {"tag": "Vercel Docs", "limit": 300},
    "docs.netlify.com": {"tag": "Netlify Docs", "limit": 300},

    # Systems & Operating Manuals
    "www.gnu.org": {"tag": "GNU Manuals", "limit": 350},
    "man7.org": {"tag": "Linux man-pages", "limit": 400},
    "wiki.archlinux.org": {"tag": "Arch Wiki", "limit": 400},
    "www.debian.org": {"tag": "Debian Docs", "limit": 350},
    "ubuntu.com": {"tag": "Ubuntu Docs", "limit": 350},
    "docs.fedoraproject.org": {"tag": "Fedora Docs", "limit": 350},
    "docs.freebsd.org": {"tag": "FreeBSD Handbook", "limit": 350},

    # Educational Resources
    "ocw.mit.edu": {"tag": "MIT OpenCourseWare", "limit": 400},
    "libretexts.org": {"tag": "LibreTexts", "limit": 400},
    "openstax.org": {"tag": "OpenStax", "limit": 300},
    "nptel.ac.in": {"tag": "NPTEL", "limit": 300}
}

class DocsSpider(scrapy.Spider):
    name = "docs_spider"
    
    # Generate seed entry points dynamically for all domains
    start_urls = [f"https://{domain}" for domain in DOCS_CONFIG.keys()]

    def __init__(self, *args, **kwargs):
        super(DocsSpider, self).__init__(*args, **kwargs)
        
        # In-memory lookups for performance
        self.scraped_urls = set()
        self.counts = {domain: 0 for domain in DOCS_CONFIG}
        
        # Load existing data file to build historical deduplication index
        dataset_path = "../../backend/data/crawler_dataset.jsonl"
        if os.path.exists(dataset_path):
            print("🔄 Loading existing dataset to prevent duplicates...")
            try:
                with open(dataset_path, "r", encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            item = json.loads(line)
                            url = item.get("url")
                            if url:
                                self.scraped_urls.add(url)
                                # Map back to the domain tracker so we don't overshoot our new limits
                                parsed_url = urlparse(url)
                                domain = parsed_url.netloc.replace("www.", "")
                                if domain in self.counts:
                                    self.counts[domain] += 1
                                    
                print(f"✅ Deduplication Engine Active: Loaded {len(self.scraped_urls)} pre-existing URLs.")
            except Exception as e:
                print(f"⚠️ Error reading existing file ({e}). Running fresh duplicate tracking.")

    def parse(self, response):
        parsed_url = urlparse(response.url)
        domain = parsed_url.netloc.replace("www.", "")
        
        if domain not in DOCS_CONFIG:
            return

        # --- LINK DISCOVERY FIRST ---
        # Even if we already saved THIS specific page, we MUST extract its links
        # so we can find other deeper pages we haven't scraped yet!
        if self.counts[domain] < DOCS_CONFIG[domain]["limit"]:
            for href in response.xpath("//a/@href").getall():
                next_page = response.urljoin(href)
                next_parsed = urlparse(next_page)
                next_domain = next_parsed.netloc.replace("www.", "")
                
                # Schedule the link if it belongs to the same site and isn't queued/scraped yet
                if next_domain == domain and next_page not in self.scraped_urls:
                    yield scrapy.Request(next_page, callback=self.parse)

        # --- DUPES & LIMIT FILTER FOR SAVING ---
        # Now we decide if we actually want to save THIS specific page to our file
        if response.url in self.scraped_urls:
            return

        if self.counts[domain] >= DOCS_CONFIG[domain]["limit"]:
            return

        text_nodes = response.xpath("//body//*[not(self::script or self::style)]/text()").getall()
        clean_text = " ".join([t.strip() for t in text_nodes if t.strip()])
        
        if len(clean_text) < 200:
            return

        # Yield item structure out to the dataset file stream
        yield {
            "id": f"crawl_{hash(response.url)}",
            "title": response.xpath("//title/text()").get(default="Documentation Page").strip(),
            "url": response.url,
            "text_content": clean_text[:4000],
            "source_type": "Documentation",
            "tags": [DOCS_CONFIG[domain]["tag"]]
        }
        
        # Track that we officially added it
        self.scraped_urls.add(response.url)
        self.counts[domain] += 1