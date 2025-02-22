import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from rank_bm25 import BM25Okapi

# Sample documents about sailing in Croatia
documents = ['Sailing in Croatia often includes visiting UNESCO World Heritage sites.',
 'The Makarska Riviera is known for its stunning coastline and sailing opportunities.',
 'Sailing in Croatia offers stunning views of the Adriatic Sea.',
 "The city of Sibenik is home to the impressive St. James's Cathedral, a UNESCO World Heritage site.",
 'The island of Brač is known for its beautiful beaches and great sailing conditions.',
 'Sailing to the island of Rab, known for its medieval old town, is a great experience.',
 'The Pakleni Islands near Hvar are a popular spot for sailing yachts.',
 'The island of Cres is one of the largest in Croatia and a great destination for sailing.',
 'The island of Cres is one of the largest in Croatia and has a diverse wildlife.',
 'The city of Zagreb is the capital of Croatia and offers a mix of modern and historic attractions.',
 'Sailors can experience the traditional Dalmatian way of life in many coastal villages.',
 'The city of Knin is known for its historic fortress and beautiful scenery.',
 'The island of Krk is the largest island in the Adriatic Sea.',
 'Croatia has a rich history dating back to the Roman Empire.',
 'The island of Hvar is a popular destination for celebrities and high-end travelers.',
 'The city of Varazdin is known for its baroque buildings and vibrant cultural scene.',
 'Sailors can enjoy fresh seafood at many coastal restaurants in Croatia.',
 'The coastal town of Senj is known for its carnival and Nehaj Fortress.',
 'The Dalmatian Coast is a famous region for sailing in Croatia.',
 "The Diocletian's Palace in Split is one of the most famous Roman ruins in Croatia.",
 'The island of Dugi Otok is known for its dramatic cliffs and beautiful sailing waters.',
 'The island of Rab is famous for its medieval old town and stunning beaches.',
 'The island of Mljet is home to a national park and is ideal for nature lovers.',
 'Sailors in Croatia can visit ancient Roman ruins in Split.',
 'The city of Kotor, just across the border in Montenegro, is a popular extension for Croatian sailing trips.',
 'The island of Korcula is believed to be the birthplace of Marco Polo.',
 'The town of Motovun in Istria is famous for its film festival.',
 'The city of Nin is known for its ancient salt pans and historic church.',
 'The city of Vukovar is known for its role in the Croatian War of Independence.',
 'Sailing to the island of Lošinj, known for its health tourism, is a relaxing experience.',
 'The island of Hvar is known for its vibrant nightlife and sailing opportunities.',
 'The island of Korčula is believed to be the birthplace of Marco Polo and is a popular sailing destination.',
 'Sailing in Croatia often involves stopping at picturesque fishing villages.',
 'The city of Opatija is known for its grand villas and seaside promenade.',
 'The city of Porec is known for the Euphrasian Basilica, a UNESCO World Heritage site.',
 'The island of Brac is home to the famous Zlatni Rat beach, known for its changing shape.',
 'Sailing around the island of Mljet offers a peaceful and scenic experience.',
 'The Blue Cave on Biševo Island is a must-see for sailors.',
 'The island of Pag is famous for its cheese and nightlife, and is a fun stop for sailors.',
 "The island of Lastovo is one of Croatia's most remote and tranquil destinations.",
 'Sailors can enjoy snorkeling in the clear waters of the Adriatic Sea.',
 'The city of Rovinj is a charming starting point for a sailing trip in Croatia.',
 "Croatia's national parks, like Krka and Plitvice, are ideal for hiking and nature lovers.",
 'The city of Zadar is famous for its sunsets, which sailors can enjoy from the sea.',
 'Croatia has over a thousand islands, each with its unique charm.',
 "Croatia's Adriatic coast is dotted with charming fishing villages.",
 'Sailing to the island of Šolta offers a quiet escape from the more touristy areas.',
 'Sailors in Croatia can explore over 1,200 islands.',
 'Sailing in Croatia provides opportunities to visit ancient fortresses and castles along the coast.',
 'Croatia is famous for its beautiful coastline and crystal-clear waters.',
 'The island of Krk is accessible by bridge and is a popular starting point for sailing trips.',
 'Sailing in Croatia is best enjoyed during the summer months.',
 'The island of Pag is famous for its cheese, which is considered a delicacy.',
 'The medieval town of Rovinj is one of the most picturesque places in Croatia.',
 'The island of Losinj is famous for its health tourism and clean air.',
 'The city of Dubrovnik, with its famous city walls, is a top destination for sailors.',
 'Croatia has beautiful islands perfect for sailing.',
 'Sailing in Croatia allows you to explore hidden coves and bays.',
 'The city of Rijeka is an important cultural and economic center in Croatia.',
 'The island of Mljet has a national park that is perfect for exploring by sailboat.',
 'The Istrian Peninsula is famous for its truffles and gourmet food.',
 'The town of Cavtat is a quieter alternative to nearby Dubrovnik.',
 'The Peljesac Peninsula is known for its vineyards and wine production.',
 'The Dubrovnik Summer Festival is a major cultural event featuring theater, music, and dance performances.',
 'The city of Zadar is famous for its unique Sea Organ, an architectural sound art object.',
 'The beaches in Croatia are among the best in Europe, with many receiving Blue Flag status.',
 'Sailing around the island of Murter gives access to the Kornati Islands National Park.',
 'The city of Trogir, with its historic architecture, is a great place to dock.',
 'Sailing from Split to Dubrovnik offers breathtaking coastal scenery.',
 'The Plitvice Lakes National Park is a UNESCO World Heritage site known for its stunning waterfalls and lakes.',
 'The city of Hvar is one of the sunniest places in Europe and a popular sailing hub.',
 "The city of Dubrovnik is often called the 'Pearl of the Adriatic'.",
 'The city of Osijek is located in the eastern part of Croatia and is known for its Baroque style.',
 'The city of Pula, with its Roman amphitheater, is a unique sailing destination.',
 'The town of Trogir is a UNESCO World Heritage site known for its medieval architecture.',
 'Sailors in Croatia can enjoy local wines at many coastal vineyards.',
 'Croatia has a Mediterranean climate, making it a great destination year-round.',
 'Croatia has a diverse cultural heritage, with influences from Italy, Hungary, and Austria.',
 'The city of Rijeka is an important cultural and historical sailing destination.',
 'The best sailing routes in Croatia include Dubrovnik and Split.',
 'The Brijuni Islands are a national park and a former presidential retreat.',
 'Sailing to the Elaphiti Islands offers a mix of natural beauty and cultural sites.',
 'Croatia is known for its delicious seafood cuisine.',
 'The ancient city of Pula is known for its well-preserved Roman amphitheater.',
 'The city of Opatija is a historical seaside resort that welcomes sailors.',
 'The waters around Croatia are known for being calm and clear, ideal for sailing.',
 'The Kornati Islands National Park is a popular sailing destination in Croatia.',
 "Sailing around the Brijuni Islands offers a glimpse of Croatia's natural beauty and wildlife.",
 'The island of Vis was a military base and was closed to tourism until the 1990s.',
 'The city of Split is a major port and gateway to the Dalmatian islands.',
 'Sailing to the island of Vis provides access to the famous Blue Cave.',
 "The city of Buzet in Istria is known as the 'City of Truffles'.",
 'The city of Karlovac is known for its parks and the rivers that flow through it.',
 'Sailing to the island of Vis allows you to experience a more remote part of Croatia.',
 'The city of Šibenik is a UNESCO World Heritage site and a great stop for sailors.',
 "The town of Samobor is famous for its traditional cream cake called 'kremsnita'.",
 'Sailing around the island of Lastovo provides a more secluded experience.',
 "Croatia's wine regions produce some excellent wines, especially in Istria and Dalmatia.",
 'Many sailors start their Croatian adventure from the city of Zadar.',
 'The coastal town of Makarska is known for its beautiful beaches and lively nightlife.']


nltk.download('punkt')
nltk.download('stopwords')
nltk.download('punkt_tab')


text = "Sailing in Croatia offers stunning views of the Adriatic Sea."
# Tokenize the text into sentences
# Splits the input text into a list of sentences
sentences = nltk.sent_tokenize(text)
print("Sentences:", sentences)


# Tokenize the text into words
# Splits the input text into a list of words and punctuation marks
words = nltk.word_tokenize(text)
print("Words:", words)


# Function to preprocess the text
def preprocess(text):
    # Tokenize the text into words and convert to lowercase
    tokens = nltk.word_tokenize(text.lower())

    # Return a list of alphanumeric words only (removing punctuation and special characters)
    return [word for word in tokens if word.isalnum()]

# The 'preprocess' function is applied to each document, and the tokens are joined back into a string
preprocessed_docs = [' '.join(preprocess(doc)) for doc in documents]


# Create an instance of TfidfVectorizer
vectorizer = TfidfVectorizer()
# Fit the vectorizer to the preprocessed documents and transform them into a TF-IDF matrix
tfidf_matrix = vectorizer.fit_transform(preprocessed_docs)


# Query the index
query = "Croatia sailing"
# Transform the query into a TF-IDF vector
query_vec = vectorizer.transform([query])


# Function to search the documents based on a query
def search(query, vectorizer, tfidf_matrix):
    # Transform the query into a TF-IDF vector using the provided vectorizer
    query_vec = vectorizer.transform([query])

    # Compute the similarity between the query vector and each document vector
    # This is done by calculating the dot product of the TF-IDF matrix and the query vector
    results = np.dot(tfidf_matrix, query_vec.T).toarray()

    # Return the array of similarity scores
    return results


# Sort and Display the results
results = search(query, vectorizer, tfidf_matrix)
sorted_results = sorted(enumerate(results), key=lambda x: x[1][0], reverse=True)


# Iterate over the sorted results and print the document with scores
for idx, score in sorted_results:
  print(f"Score {score[0]:.2f} Document {documents[idx]}") # Use idx (integer) to access document


# Boolean Retrieval --------------------------------------------------------------


# Process the text: tokenization, only alphanumeric, lower case, no stopwords except logical operators
def process_text(text):
    # Convert text to lower case
    processed_text = text.lower()

    # Tokenize the text
    processed_text = nltk.word_tokenize(processed_text)

    # Filter out non-alphanumeric tokens
    processed_text = [word for word in processed_text if word.isalnum()]

    # Define stopwords while excluding logical operators
    stopwords = set(nltk.corpus.stopwords.words("english")) - {"and", "or", "not"}

    # Filter out stopwords except logical operators
    processed_text = [word for word in processed_text if word not in stopwords]

    return processed_text


# Simplified Boolean Retrieval System to handle the 'NOT' operator correctly
def boolean_search(query, documents):
    # Preprocess the query and split it into tokens
    query_tokens = process_text(query)

    # Initialize an empty list to store the matching results
    results = []

    # Iterate over each document in the collection
    for doc_id, doc in enumerate(documents):
        # Preprocess the current document and convert it into a set of unique tokens
        doc_tokens = set(process_text(doc))

        # Initialize a flag to determine if the document should be included in the results
        include_doc = False

        # Iterate through each token in the query
        for i, token in enumerate(query_tokens):
            # Handle the 'AND' operator
            if token == "and" and i + 1 < len(query_tokens):
                # If the 'AND' operator is found, check if the next token is in the document
                include_doc = include_doc and (query_tokens[i + 1] in doc_tokens)
            # Handle the 'OR' operator
            elif token == "or" and i + 1 < len(query_tokens):
                # If the 'OR' operator is found, include the document if the next token is present
                include_doc = include_doc or (query_tokens[i + 1] in doc_tokens)
            # Handle the 'NOT' operator
            elif token == "not" and i + 1 < len(query_tokens):
                # If the 'NOT' operator is found, exclude the document if the next token is present
                if query_tokens[i + 1] in doc_tokens:
                    include_doc = False
                    break  # Exit the loop early as this document should be excluded
            else:
                # For normal tokens (i.e., not operators), check if the token is in the document
                # If this is the first token, it determines the initial inclusion status
                # If it's subsequent, it combines with previous logic using 'OR'
                include_doc = token in doc_tokens or include_doc

        # After processing all tokens in the query, check if the document should be included
        if include_doc:
            # If the document matches the query, add it to the results list
            results.append((doc_id, doc))

    # Return the list of documents that match the query
    return results


# Define the search query
query = "sailing not croatia"

# Perform a boolean search on the documents using the query
boolean_results = boolean_search(query, documents)

# Print the results of the boolean search
print("Boolean Search Results for query '{}':".format(query))

# Iterate through the search results and print the document ID and content
for doc_id, doc in boolean_results:
    print("Document ID: {}, Document: {}".format(doc_id, doc))


# Probabilistic Retrieval --------------------------------------------------------------

# Tokenize the documents for BM25
tokenized_docs = [process_text(doc) for doc in documents]

# Initialize BM25 model
bm25 = BM25Okapi(tokenized_docs)

# Function to perform a BM25 search on the documents
def bm25_search(query, bm25):
    # Process the query text
    query_tokens = process_text(query)

    # Get the BM25 scores for the query against the indexed documents
    results = bm25.get_scores(query_tokens)

    # Return the BM25 scores
    return results


# Perform BM25 search to get relevance scores for the query
bm25_results = bm25_search(query, bm25)


# Sort the results in descending order of relevance scores
sorted_bm25_results = np.argsort(bm25_results)[::-1]

# Display the sorted BM25 search results
print("\nBM25 Search Results for query '{}':".format(query))
for idx in sorted_bm25_results:
    print("Score: {:.4f}, Document: {}".format(bm25_results[idx], documents[idx]))