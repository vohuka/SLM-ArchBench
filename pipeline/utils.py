"""
Utility functions with improved architectural component extraction.
"""
import re
from typing import Set


def count_parameters(model):
    """Count total and trainable parameters in the model."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total_params_M": round(total_params / 1e6, 2),
        "trainable_params_M": round(trainable_params / 1e6, 2),
        "trainable_percentage": round(100 * trainable_params / total_params, 2)
    }


def extract_components(text: str) -> Set[str]:
    """
    Extract architectural components with relaxed rules for ADR domain.
    Handles both concrete (file names, classes) and abstract (concepts, tech stack) components.
    """
    components = set()
    
    # ============================================================
    # TIER 1: Concrete code artifacts (HIGH PRECISION)
    # ============================================================
    
    # Pattern 1: File names
    file_pattern = r'\b\w+\.(java|py|ts|tsx|jsx|js|xml|yaml|yml|json|properties|conf|kt|scala|rs|go|rb|hs|ex|elixir)\b'
    files = re. findall(file_pattern, text, re.IGNORECASE)
    components.update(files)
    
    # Pattern 2: Class/Service names with architectural suffixes
    class_pattern = r'\b[A-Z][a-zA-Z0-9]*(?:Service|Controller|Repository|Model|Manager|Handler|Provider|Factory|Builder|Adapter|Facade|Strategy|Component|Module|Constraint|Interval|Config|Validator)\b'
    classes = re. findall(class_pattern, text)
    components.update(classes)
    
    # Pattern 3: Package/Module names (e.g., com.example.payment)
    package_pattern = r'\b[a-z]+(?:\.[a-z]+){2,}\b'
    packages = re.findall(package_pattern, text)
    components.update(packages)
    
    # Pattern 4: Architecture layer keywords
    layer_keywords = r'\b(controller|service|repository|model|view|database|api|frontend|backend|middleware|gateway|proxy|cache|queue|ledger|blockchain|datatype|constraint|library)\b'
    layers = re. findall(layer_keywords, text, re.IGNORECASE)
    components.update([l.lower() for l in layers])
    
    # ============================================================
    # TIER 2: Technology stack & frameworks (MEDIUM PRECISION)
    # ============================================================
    
    # Known tech stack (curated whitelist)
    tech_stack = {
        # Cloud & Infrastructure
        'AWS', 'Azure', 'GCP', 'Google Cloud', 'Heroku', 'DigitalOcean', 'Kubernetes', 'Docker',
        # Languages
        'Python', 'Java', 'JavaScript', 'TypeScript', 'Kotlin', 'Scala', 'Go', 'Rust', 'Ruby', 'PHP', 
        'C#', 'C++', 'Haskell', 'Elixir', 'Clojure',
        # Frontend Frameworks
        'React', 'Angular', 'Vue', 'Svelte', 'Next.js', 'Nuxt', 'Gatsby', 'Bootstrap', 'Tailwind', 
        'Material-UI', 'Bulma', 'Foundation', 'Semantic UI', 'BEM',
        # Backend Frameworks
        'Django', 'Flask', 'FastAPI', 'Spring', 'Hibernate', 'Express', 'NestJS', 'Laravel', 'Rails', 
        'ASP.NET', 'Rocket', 'Actix',
        # Databases
        'PostgreSQL', 'MySQL', 'MongoDB', 'Redis', 'Cassandra', 'DynamoDB', 'Elasticsearch', 'Neo4j',
        # Message queues & Event streaming
        'Kafka', 'RabbitMQ', 'SQS', 'SNS', 'Pulsar', 'NATS',
        # Architecture patterns
        'MVC', 'MVVM', 'Microservices', 'Monolith', 'Serverless', 'Event-Driven', 'CQRS', 'REST', 
        'GraphQL', 'gRPC', 'SOAP', 'TOSCA', 'ADR',
        # DevOps & CI/CD
        'Jenkins', 'GitLab', 'GitHub Actions', 'CircleCI', 'Travis CI', 'Terraform', 'Ansible', 'Puppet', 'Chef',
        # Blockchain specific (for your dataset)
        'Cardano', 'Plutus', 'Marconi', 'Ethereum', 'Bitcoin', 'Solidity', 'Smart Contract',
        # Build tools
        'Maven', 'Gradle', 'Webpack', 'Vite', 'Rollup', 'Parcel', 'npm', 'yarn', 'pnpm',
        # Testing
        'Jest', 'Mocha', 'Pytest', 'JUnit', 'Selenium', 'Cypress', 'Playwright',
        # Version control
        'Git', 'GitHub', 'GitLab', 'Bitbucket', 'SVN',
        # Monitoring
        'Prometheus', 'Grafana', 'Datadog', 'New Relic', 'Splunk',
    }
    
    # Case-insensitive matching with word boundaries
    for tech in tech_stack:
        # Use word boundary to avoid partial matches
        pattern = r'\b' + re. escape(tech) + r'\b'
        if re.search(pattern, text, re.IGNORECASE):
            components.add(tech)
    
    # ============================================================
    # TIER 3: Repository & project names (NEW - for your dataset)
    # ============================================================
    
    # Pattern for GitHub-style repo names (e.g., "input-output-hk/cardano-world")
    repo_pattern = r'\b[a-zA-Z0-9-]+/[a-zA-Z0-9-]+\b'
    repos = re.findall(repo_pattern, text)
    components.update(repos)
    
    # Pattern for project/library names (e.g., "plutus-apps", "snake-yaml")
    project_pattern = r'\b[a-z]+-[a-z]+(? :-[a-z]+)*\b'
    projects = re. findall(project_pattern, text)
    # Filter out common false positives
    project_blacklist = {'open-source', 'well-known', 'up-to-date', 'user-friendly', 'real-time'}
    components.update([p for p in projects if p not in project_blacklist and len(p) > 5])
    
    # ============================================================
    # TIER 4: CamelCase identifiers (type names, function names)
    # ============================================================
    
    # Pattern for CamelCase identifiers (likely type/function names)
    camel_case_pattern = r'\b[A-Z][a-z]+(?:[A-Z][a-z]+)+\b'
    camel_identifiers = re.findall(camel_case_pattern, text)
    
    # Whitelist: only include if length > 8 OR appears multiple times
    for identifier in camel_identifiers:
        if len(identifier) > 8 or text.count(identifier) >= 2:
            components.add(identifier)
    
    # ============================================================
    # TIER 5: Proper nouns with strict filtering (LOW PRECISION - use carefully)
    # ============================================================
    
    # Extract capitalized words
    proper_noun_pattern = r'\b[A-Z][a-zA-Z0-9]+\b'
    potential_components = re.findall(proper_noun_pattern, text)
    
    # Comprehensive blacklist
    blacklist = {
        # English articles & pronouns
        'The', 'A', 'An', 'This', 'That', 'These', 'Those', 'It', 'I', 'We', 'You', 'They', 'He', 'She',
        # Prepositions & conjunctions
        'In', 'On', 'At', 'To', 'For', 'With', 'By', 'From', 'As', 'Of', 'And', 'Or', 'But', 'If', 'When',
        # ADR-specific words
        'Architecture', 'Decision', 'Record', 'Context', 'Outcome', 'Status', 'Date', 'Title', 'Rationale',
        'Alternative', 'Alternatives', 'Considered', 'Chosen', 'Example', 'Explanation', 'Reasoning',
        # Verbs & modals
        'Given', 'When', 'Then', 'Should', 'Must', 'Could', 'Would', 'Can', 'May', 'Will', 'Shall',
        # Generic terms
        'System', 'Application', 'Software', 'Framework', 'Library', 'Tool', 'Platform', 'Solution',
        # Others
        'However', 'Therefore', 'Thus', 'Hence', 'Furthermore', 'Moreover', 'Additionally', 'Instead',
    }
    
    for word in potential_components:
        if (word not in blacklist and 
            len(word) > 3 and  # Avoid short acronyms
            word not in components and  # Avoid duplicates
            (text.count(word) >= 2 or  # Appears multiple times
             any(keyword in text. lower() for keyword in ['system', 'service', 'platform', 'framework', 'tool', 'infrastructure']))):
            components.add(word)
    
    return components