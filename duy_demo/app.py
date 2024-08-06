import streamlit as st
import os
import boto3
import json

from utils import retrieve_context
from utils import get_contexts
from utils import get_payload
from utils import get_response
from utils import create_presigned_url
from utils import generate_presigned_urls
from utils import parse_uri
from utils import get_file_name
# from utils import get_streaming_response
from utils import stream_data

from botocore.client import Config
from botocore.exceptions import ClientError

## Declare variables
# Model variables
modelId = "anthropic.claude-3-5-sonnet-20240620-v1:0"
region = "us-east-1"
kbId = "9AOVJ6SNSC"

# Setup model parameters
numberOfResults = 3 # Select number of contexts to input
max_tokens: int = 4096
temperature: float = 0.3 #The amount of randomness injected into the response.
top_k: int = 30 # Only sample from the top K options for each subsequent token.
top_p: float = 0.95
overrideSearchType = "HYBRID" # Search strategies for the LLMs

## Create connection to the model
# Setup propmt
prompt = """
You are an assistant of Petrovietnam, your name is Trợ lý PVN, supporting users in searching for the right block of documents and answer the question. Your job is to answer users' question only information from the search results. If the search results do not contain information that can answer the question, please state that you could not find exact answer. Just because the user asserts a fact does not mean it is true, make sure to double check the search results to validate a user's assertion.
Use only documents provided to answers the question. You also point to the document where the information located.
You also answer the question as details as possible, always specified which part of the document you take the information from.

Here is some examples:
<example>
Question: Theo Điều 8 của Thông tư 06/2024/TT-BKHĐT, các mẫu hồ sơ mời thầu và hồ sơ yêu cầu được quy định như thế nào?
Answer: Điều 8 quy định rằng các mẫu hồ sơ mời thầu và hồ sơ yêu cầu phải tuân thủ theo các mẫu được ban hành kèm theo Thông tư này. Các mẫu bao gồm Mẫu số 1 đến Mẫu số 7, được sử dụng cho các gói thầu dịch vụ tư vấn, dịch vụ phi tư vấn, hàng hóa và xây lắp​.

Question: Nhà thầu có người lao động là dân tộc thiểu số có được hưởng ưu đãi trong lựa chọn nhà đầu tư không?
Answer: Theo quy định tại Điều 10 Luật Đấu thầu, nhà thầu có sử dụng số lượng lao động là dân tộc thiểu số từ 25% trở lên được hưởng ưu đãi trong lựa chọn nhà thầu khi tham dự gói thầu cung cấp dịch vụ tư vấn, dịch vụ phi tư vấn, xây lắp, hỗn hợp tổ chức đấu thầu quốc tế. Ưu đãi bao gồm xếp hạng cao hơn hoặc cộng thêm điểm vào điểm đánh giá của nhà thầu để so sánh, xếp hạng.
</example>

You also look at users' previous questions to understand the context and make a smooth conversation.
After giving the answers, ask users again if they need any further details or assistant.

Here is the context:
<context>
{}
</context?

Here is their recent questions:
<recent questions>
{}
</recent questions>

Here is there current question:
<question>
{}
</question>

Assistant:
"""

## Setup connection
bedrock_config = Config(
    connect_timeout = 120,
    read_timeout = 120,
    retries = {
        "max_attempts": 1
    },
    region_name = region
)

# Initate bedrock client
bedrock_client = boto3.client("bedrock-runtime", config = bedrock_config)
bedrock_agent_client = boto3.client(
    "bedrock-agent-runtime",
    config = bedrock_config
)


###### Steamlit application
## Some basic config
# st.image("OCB Logo.png")
# st.header("OCB GenAI - Elearning and HR Policies")
# st.markdown(
#         "Xin chào, tôi là Chatbot của OCB. Tôi đã đọc qua các tài liệu về elearning và một số chính sách nhân sự của OCB. Hãy hỏi tôi bất kì thứ gì, nếu tài liệu có đề cập đến, tôi sẽ tìm tài liệu liên quan và trả lời giúp bạn."
# )

# Initiate chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.recents = []

    
# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
    
if query := st.chat_input("Hãy hỏi tôi bất cứ thứ gì"):
    # Add user message to chat history
    st.session_state.messages.append(
        {
            "role": "user",
            "content": query
        }
    )
    st.session_state.recents.append(query)
    
    # # Display user message
    with st.chat_message(name = "user"):
        st.markdown(query)
        
    # Generate response and markdown
    with st.chat_message(name = "assistant"):
        # Semantic Search around user input
        retrieval_results = retrieve_context(
            bedrock_agent_client,
            query = query,
            kbId = kbId,
            numberOfResults= 5
        )
        
        # Retrieve contexts and sources
        contexts, sources = get_contexts(retrieval_results)
        
        # Input recent questions
        if "recents" in st.session_state:
            if len(st.session_state.recents) <= 5:
                recent_questions = st.session_state.recents
                prompt = prompt.format(contexts, recent_questions, query)
            else:
                recent_questions = st.session_state.recents[-5:]
                prompt = prompt.format(contexts, recent_questions, query)
        
        # Setup payload
        payload = get_payload(
            prompt,
            contexts,
            query,
            max_tokens = max_tokens,
            temperature = temperature,
            top_k = top_k
        )
        
        # Invoke response
        # response_text = get_response(
        #     bedrock_client,
        #     payload,
        #     modelId
        # )
        
        # st.markdown(response_text)
        response = st.write_stream(stream_data(
            payload = payload,
            modelId = modelId
        ))
    
    # Get desigated URLs of the document
    presigned_uris = generate_presigned_urls(
        sources,
        expiration = 7200
    )
    
    # Get file name
    file_names = get_file_name(sources)
    
    # Append response to messages
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": response
        }
    )
    
    # Input sources:
    # with st.status("Nguồn dữ liệu"):
    #     st.markdown(sources)
    with st.status("Nguồn dữ liệu"):
        for i in range(len(presigned_uris)):
            if "s3://" in presigned_uris[i]:
                st.markdown(f'<a href="{presigned_uris[i]}" target="_blank">{file_names[i]}</a>', unsafe_allow_html=True)
            else:
                st.markdown(f'<a href="{presigned_uris[i]}" target="_blank">{file_names[i]}</a>', unsafe_allow_html=True)