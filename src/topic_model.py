from bertopic import BERTopic
from bertopic.vectorizers import ClassTfidfTransformer
from keybert import KeyBERT
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer


def create_embeddings(df, sent_transformer="all-MiniLM-L6-v2"):
    sentence_model = SentenceTransformer(sent_transformer)
    summary_docs = df.dropna(subset=["text"]).summary.values.tolist()
    embeddings = sentence_model.encode(summary_docs, show_progress_bar=True)
    return summary_docs, embeddings


def get_keywords(docs):
    kw_model = KeyBERT()
    keywords = kw_model.extract_keywords(docs)
    return keywords


def get_vocabulary(keywords):
    vocabulary = [k[0] for keyword in keywords for k in keyword]
    vocabulary = list(set(vocabulary))
    return vocabulary


def build_topic_model(vocabulary):
    ctfidf_model = ClassTfidfTransformer(reduce_frequent_words=True)
    vectorizer_model = CountVectorizer(stop_words="english", vocabulary=vocabulary)
    model = BERTopic(
        language="english",
        vectorizer_model=vectorizer_model,
        diversity=0.2,
        min_topic_size=10,
        ctfidf_model=ctfidf_model,
        top_n_words=15,
    )
    return model


def create_topic_model(df):
    summary_docs, embeddings = create_embeddings(df)
    keywords = get_keywords(docs=summary_docs)
    vocabulary = get_vocabulary(keywords)
    model = build_topic_model(vocabulary)

    topics, probs = model.fit_transform(summary_docs, embeddings)
    num_classified = model.get_topic_freq().Count.values[1:].sum()
    print(f"classified {num_classified} out of {len(summary_docs)}")
    return {"topics": topics, "probs": probs, "model": model}
