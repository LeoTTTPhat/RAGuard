from setuptools import setup, find_packages

setup(
    name="ragentguard",
    version="0.1.0",
    description="End-to-End Security and Privacy Auditing for RAG-Agent Pipelines",
    author="RAGentGuard Research Team",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.10",
    extras_require={
        "langchain": ["langchain>=0.2.0", "chromadb>=0.5.0"],
        "llm-attribution": ["torch>=2.2.0", "transformers>=4.40.0"],
        "eval": ["datasets>=2.20.0", "numpy>=1.26.0", "pandas>=2.2.0"],
        "neural-eval": [
            "sentence-transformers>=3.0.0",
            "faiss-cpu>=1.8.0",
            "chromadb>=0.5.0",
            "torch>=2.2.0",
            "transformers>=4.40.0",
        ],
        "dev": ["pytest>=8.0.0", "pytest-cov>=5.0.0"],
    },
)
