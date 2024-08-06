import streamlit as st
import boto3
import json

from botocore.client import Config
from botocore.exceptions import ClientError

## Declare variable
modelId = "anthropic.claude-3-5-sonnet-20240620-v1:0"
region = "us-east-1"
kbId = "9AOVJ6SNSC"

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

## Define handful function
# Define retrieve context
def retrieve_context(bedrock_agent_client,query, kbId, numberOfResults = 5):
    response = bedrock_agent_client.retrieve(
        retrievalQuery = {
            "text": query
        },
        knowledgeBaseId = kbId,
        retrievalConfiguration = {
            "vectorSearchConfiguration": {
                "numberOfResults": numberOfResults,
                "overrideSearchType": "HYBRID"
            }
        }
    )
    
    retrieval_results = response["retrievalResults"]
    return retrieval_results

# Fetch contents from the response
def get_contexts(retrieval_results):
    # Initiate list of contexts and sources
    contexts = []
    sources = []
    
    # Loop through the sources 
    for i in range(len(retrieval_results)):
        if retrieval_results[i]['location']['type'] == "WEB":
            contexts.append(retrieval_results[i]['content']['text'])
            sources.append(retrieval_results[i]['metadata']['x-amz-bedrock-kb-source-uri'])
        else:
            contexts.append(retrieval_results[i]['content']['text'])
            sources.append(retrieval_results[i]['location']['s3Location']['uri'])
            
    # Get the unique sources:
    sources = list(set(sources))
    
    return contexts, sources

## Invoke LLMs models from Amazon Bedrock
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
# Get payload
def get_payload(
    prompt,
    contexts,
    query,
    max_tokens,
    temperature: float,
    top_k: int,
    top_p: float = None,
):
    # Define message
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type":"text",
                    "text": prompt
                }
            ]
        }
    ]
    
    # Define payload
    if top_p is not None:
        sonnet_payload = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": max_tokens,
            "messages": messages,
            "top_p": top_p,
            "top_k": top_k
        }
        sonnet_payload = json.dumps(sonnet_payload)
        return sonnet_payload
    else:
        sonnet_payload = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": max_tokens,
            "messages": messages,
            "temperature": temperature,
            "top_k": top_k
        }
        sonnet_payload = json.dumps(sonnet_payload)
        return sonnet_payload
    
def get_response(bedrock_client, payload, modelId):
    response = bedrock_client.invoke_model(
        body = payload,
        modelId = modelId,
        accept = "application/json",
        contentType = "application/json"
    )
    
    response_body = json.loads(
        response.get("body").read()
    )
    
    response_text = response_body.get("content")[0]["text"]
    return response_text
import re
def get_web_name(uri):
    # Remove the protocol (http:// or https://)
    if uri.startswith("http://"):
        uri = uri[7:]
    elif uri.startswith("https://"):
        uri = uri[8:]
    
    # Remove the file extension
    uri = re.sub(r'\.[a-zA-Z]+$', '', uri)
    
    return uri

# Get link for the URI
def parse_uri(uri):
    if uri.startswith("s3://"):
        uri = uri[5:]
        bucket_name, object_key = uri.split("/", 1)
        return "s3", bucket_name, object_key
    elif uri.startswith("http://") or uri.startswith("https://"):
        return "web", uri, None
    else:
        raise ValueError("Invalid URI")

def create_presigned_url(bucket_name, object_name, expiration=7200):
    s3_client = boto3.client("s3")
    try:
        response = s3_client.generate_presigned_url(
            "get_object",
            Params={"Bucket": bucket_name, "Key": object_name},
            ExpiresIn=expiration
        )
    except ClientError as e:
        logging.error(e)
        return None
    return response

def generate_presigned_urls(uris, expiration=7200):
    presigned_urls = []
    for uri in uris:
        uri_type, bucket_or_url, object_key = parse_uri(uri)
        if uri_type == "s3":
            presigned_url = create_presigned_url(bucket_or_url, object_key, expiration)
            if presigned_url:
                presigned_urls.append(presigned_url)
        elif uri_type == "web":
            presigned_urls.append(bucket_or_url)
    return presigned_urls

def get_file_name(uris):
    file_names = []
    for uri in uris:
        if uri.startswith("s3://"):
            file_name = uri.split("//")[-1].split("/")[-1]
        elif uri.startswith("http://") or uri.startswith("https://"):
            file_name = uri.split("/")[-1]
        else:
            raise ValueError("Invalid URI")
        file_names.append(file_name)
    return file_names
# Define get streaming response
def stream_data(
    payload,
    modelId
):
    response = bedrock_client.invoke_model_with_response_stream(
        modelId = modelId,
        body = payload
    )
    
    for event in response.get("body"):
        chunk = json.loads(event["chunk"]["bytes"])
        
        if chunk["type"] == "content_block_delta" and chunk["delta"]["type"] == "text_delta":
            yield chunk["delta"]["text"]


# def get_streaming_response(payload, modelId):
#     response = bedrock_client.invoke_model_with_response_stream(
#         modelId=modelId,
#         body=payload
#     )
    
#     buffer = ""
    
#     for event in response.get("body"):
#         chunk = json.loads(event["chunk"]["bytes"])
        
#         if chunk["type"] == "content_block_delta" and chunk["delta"]["type"] == "text_delta":
#             # Append the new text to the buffer
#             buffer += chunk["delta"]["text"]
            
#             # Display the buffered text so far
#             st.markdown(buffer)
            
#             # Add the chunk to the session state messages
#             st.session_state.messages.append(
#                 {
#                     "role": "assistant",
#                     "content": chunk
#                 }
#             )