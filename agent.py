from langchain_community.llms import Ollama


prompt = "Write a short story about a robot learning to dance."


llm = Ollama(model="ollama/llama2:7b")

def classify_sentiment(title):
    output = llm.invoke(f"Classify the sentiment as 'POSITIVE' or 'NEGATIVE' or 'NEUTRAL' with just that one word only, no additional words or reasoning: {title}")
    return output.strip()  # Ensure the response is clean and without extra spaces

# Function to get and process news data
def get_news_data(ticker):

    # Data Pull
    stock = finvizfinance(ticker) 
    news_df = stock.ticker_news()

    # Preprocess before putting into LLM
    news_df['Title'] = news_df['Title'].str.lower()

    # Classify Sentiment function applied to each row of news_df
    news_df['sentiment'] = news_df['Title'].apply(classify_sentiment)

    # Postprocess after putting into LLM
    news_df['sentiment'] = news_df['sentiment'].str.upper()
    news_df = news_df[news_df['sentiment'] != 'NEUTRAL']
    news_df['Date'] = pd.to_datetime(news_df['Date'])
    news_df['DateOnly'] = news_df['Date'].dt.date

    return news_df

def chat(prompt):
    response = model.chat(prompt)
    return response

def response_to_text(response):
    return response["response"]

def fetch_finance_news():

    prompt = "Fetch the latest finance news."
    response = chat(prompt)
    return response_to_text(response)