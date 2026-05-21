use crate::Result;
use crate::error::Error::{self, MissingHeader};
use crate::hash::gen_request_hash;
use crate::model::ChatRequest;
use crate::process::ChatProcess;
use crate::serve::AppState;
use axum::{
    Json,
    extract::State,
    response::{IntoResponse, Response},
};
use axum_extra::{
    TypedHeader,
    extract::WithRejection,
    headers::{Authorization, authorization::Bearer},
};
use reqwest::{Client, header};

const ORIGIN_API: &str = "https://duck.ai";

pub async fn models(
    State(state): State<AppState>,
    bearer: Option<TypedHeader<Authorization<Bearer>>>,
) -> crate::Result<Response> {
    state.valid_key(bearer)?;

    let model_data = vec![
        serde_json::json!({
            "id": "gpt-4o-mini",
            "object": "model",
            "created": 1686935002,
            "owned_by": "openai",
        }),
        serde_json::json!({
            "id": "gpt-5-mini",
            "object": "model",
            "created": 1686935002,
            "owned_by": "openai",
        }),
        serde_json::json!({
            "id": "gpt-oss-120b",
            "object": "model",
            "created": 1686935002,
            "owned_by": "openai",
        }),
        serde_json::json!({
            "id": "llama-4-scout",
            "object": "model",
            "created": 1686935002,
            "owned_by": "meta",
        }),
        serde_json::json!({
            "id": "claude-3.5-haiku",
            "object": "model",
            "created": 1686935002,
            "owned_by": "claude",
        }),
        serde_json::json!({
            "id": "mixtral-small-3",
            "object": "model",
            "created": 1686935002,
            "owned_by": "mistral",
        }),
    ];

    Ok(Json(serde_json::json!({
        "object": "list",
        "data": model_data,
    }))
    .into_response())
}

pub async fn chat_completions(
    State(state): State<AppState>,
    bearer: Option<TypedHeader<Authorization<Bearer>>>,
    WithRejection(Json(mut body), _): WithRejection<Json<ChatRequest>, Error>,
) -> crate::Result<Response> {
    state.valid_key(bearer)?;
    let mut token = None;
    for _ in 0..5 {
        match load_token(&state.client).await {
            Ok(new_token) => {
                token = Some(new_token);
                break;
            }
            Err(err) => {
                tracing::info!("retry load token: {:?}", err);
                tokio::time::sleep(std::time::Duration::from_secs(1)).await;
            }
        }
    }
    let token = token.ok_or_else(|| Error::BadRequest("cannot get token".to_string()))?;
    // body.compress_messages();
    body.to_duck_chat_request();
    let (_, response) = send_request(&state.client, token, &body).await?;
    Ok(response)
}

async fn send_request(
    client: &Client,
    hash: String,
    body: &ChatRequest,
) -> Result<(String, Response)> {
    dbg!(&body);
    let mut body = body.clone();
    body.reasoning_effort = Some("minimal".into());
    let resp = client
        .post("https://duck.ai/duckchat/v1/chat")
        .header(header::ACCEPT, "text/event-stream")
        .header(header::ORIGIN, ORIGIN_API)
        .header(header::REFERER, ORIGIN_API)

        // .header(header::CONTENT_TYPE, "application/json")
        // .header("pragma", "no-cache")
        .header("priority", "u=1, i")
        .header("sec-fetch-dest", "empty")
        .header("sec-fetch-mode", "cors")
        .header("sec-fetch-site", "same-origin")
        .header("x-fe-version", "serp_20250401_100419_ET-19d438eb199b2bf7c300")
        .header(header::USER_AGENT, "Mozilla/5.0 (X11; Linux x86_64; rv:149.0) Gecko/20100101 Firefox/149.0")
        
        .header("x-vqd-hash-1", hash)
        .json(&body)
        .send()
        .await?;

    let hash = resp
        .headers()
        .get("x-vqd-hash-1")
        .and_then(|header| header.to_str().ok())
        .ok_or_else(|| MissingHeader)?
        .to_owned();

    let response = ChatProcess::builder()
        .resp(resp)
        .stream(body.stream)
        .model(body.model.clone())
        .build()
        .into_response()
        .await?;

    Ok((hash, response))
}

async fn load_token(client: &Client) -> Result<String> {
    let resp = client
        .get("https://duck.ai/duckchat/v1/status")
        .header(header::REFERER, ORIGIN_API)
        .header("x-vqd-accept", "1")
        .send()
        .await?
        .error_for_status()?;

    let hash = resp
        .headers()
        .get("x-vqd-hash-1")
        .and_then(|header| header.to_str().ok())
        .ok_or_else(|| crate::Error::MissingHeader)?
        .to_owned();

    let request_hash = gen_request_hash(&hash)?;

    Ok(request_hash)
}


// #[tokio::test]
// async fn my_async_test() {
//     let client = reqwest::Client::new();
//     let hash = load_token(&client).await;
//     dbg!(hash);

//     // assert_eq!(value, 4);
// }

#[cfg(test)]
mod live_tests {
    use super::*;
    use reqwest::Client;
    use crate::model::{ChatRequest, Content, Message, Role};
    use tokio::time::{timeout, Duration};

    // Increase timeout for real network calls
    const TEST_TIMEOUT_SECS: u64 = 30;

    #[tokio::test]
    async fn test_load_token_real() {
        let client = Client::new();
        let res = timeout(Duration::from_secs(TEST_TIMEOUT_SECS), load_token(&client)).await;
        let hash = res.expect("timeout when calling load_token").expect("load_token failed");
        assert!(!hash.is_empty(), "returned hash should not be empty");
    }

    #[tokio::test]
    async fn test_send_request_real() {
        // Build a minimal ChatRequest appropriate for the service.
        let body = ChatRequest {
            model: "gpt-5-mini".to_string(),
            stream: Some(false),
            messages: vec![
                Message {
                    role: Some(Role::User),
                    content: Some(Content::Text("Hello from unit test".to_string())),
                }
            ],
            compressed: false,
            reasoning_effort: None,
        };

        let client = Client::new();

        // first obtain token/hash
        let token = timeout(Duration::from_secs(TEST_TIMEOUT_SECS), load_token(&client))
            .await
            .expect("timeout when calling load_token")
            .expect("load_token failed");

        // call send_request and ensure it returns a non-empty hash and a successful response
        // let (returned_hash, response) =
            let shit = timeout(Duration::from_secs(TEST_TIMEOUT_SECS), send_request(&client, token.clone(), &body))
                .await;
            dbg!(shit);
                // .expect("timeout when calling send_request")
                // .expect("send_request failed");

        // assert!(!returned_hash.is_empty(), "returned hash should not be empty");
        // Check response is an HTTP response with success-like status (ChatProcess may already translate; check by attempting to convert into a status if accessible)
        // Here we rely on IntoResponse result being produced; ensure some bytes can be extracted by attempting to serialize to bytes.
        // If `response` is axum::response::Response, we can at least assert headers exist.
        // let _ = response.headers();
    }
}

