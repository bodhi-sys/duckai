use serde::{Deserialize, Deserializer, Serialize};
use typed_builder::TypedBuilder;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    System,
    Assistant,
    User,
}

// ==================== Request Body ====================
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ChatRequest {
    #[serde(deserialize_with = "deserialize_model")]
    pub model: String,
    #[serde(deserialize_with = "deserialize_message")]
    pub messages: Vec<Message>,
    #[serde(skip_serializing, default)]
    pub stream: Option<bool>,
    #[serde(skip_serializing, default)]
    pub compressed: bool,
    #[serde(rename="reasoningEffort", skip_serializing_if = "Option::is_none", default)]
    pub reasoning_effort: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default, TypedBuilder)]
pub struct Message {
    #[builder(default, setter(into))]
    pub role: Option<Role>,
    #[builder(default, setter(into))]
    pub content: Option<Content>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum Content {
    Text(String),
    Vec(Vec<ContentItem>),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContentItem {
    #[serde(rename = "type")]
    r#type: String,
    pub text: String,
}

fn deserialize_model<'de, D>(deserializer: D) -> Result<String, D::Error>
where
    D: Deserializer<'de>,
{
    let model = String::deserialize(deserializer)?;
    let model = match model.as_str() {
        "gpt-4o-mini" => "gpt-4o-mini",
        "gpt-5-mini" => "gpt-5-mini",
        "gpt-oss-120b" => "openai/gpt-oss-120b",
        "llama-4-scout" => "meta-llama/Llama-4-Scout-17B-16E-Instruct",
        "claude-3.5-haiku" => "claude-3-5-haiku-latest",
        "mixtral-small-3" => "mistralai/Mistral-Small-24B-Instruct-2501",
        _ => model.as_str(),
    };
    Ok(model.to_owned())
}

fn deserialize_message<'de, D>(deserializer: D) -> Result<Vec<Message>, D::Error>
where
    D: Deserializer<'de>,
{
    let mut message: Vec<Message> = Vec::deserialize(deserializer)?;
    for message in &mut message {
        if let Some(role) = message.role.as_mut() {
            if matches!(role, Role::System) {
                *role = Role::User;
            }
        }
    }
    Ok(message)
}

pub fn compress_messages(messages: &[Message]) -> String {
    let mut key = String::new();
    for message in messages {
        if let (Some(role), Some(msg)) = (&message.role, &message.content) {
            let role = serde_json::to_string(&role).unwrap();
            let role = role.trim_matches('"');
            match msg {
                Content::Text(msg) => key.push_str(&format!("{role}:{msg};\n")),
                Content::Vec(vec) => {
                    for item in vec {
                        key.push_str(&format!("{role}:{};\n", item.text));
                    }
                }
            }
        }
    }
    key
}

/// Extracts file sources from message content and returns (clean_content, Vec<(file_path, file_content)>)
pub fn extract_file_sources(content: &str) -> (String, Vec<(String, String)>) {
    use regex::Regex;

    let file_pattern = Regex::new(r"============ FILE: (.+?) ============\n").unwrap();
    let mut files = Vec::new();

    // Find all file markers and extract content
    let mut segments: Vec<(usize, usize, String, String)> = Vec::new(); // (start, end, filepath, content)

    for cap in file_pattern.captures_iter(content) {
        let full_match = cap.get(0).unwrap();
        let file_path = cap.get(1).unwrap().as_str().to_string();
        let start = full_match.end();

        // Find the next file marker or end of string
        let next_marker = file_pattern.find_at(content, start);
        let end = next_marker.map(|m| m.start()).unwrap_or(content.len());

        let file_content = content[start..end].trim().to_string();
        segments.push((full_match.start(), end, file_path, file_content));
    }

    // Build clean content by removing file sections
    let mut result = String::new();
    let mut last_pos = 0;

    for (start, end, file_path, file_content) in segments {
        // Add content before this file marker
        result.push_str(&content[last_pos..start]);
        files.push((file_path, file_content));
        last_pos = end;
    }

    // Add remaining content
    result.push_str(&content[last_pos..]);

    // Clean up extra whitespace
    let result = result.trim().to_string();

    (result, files)
}

impl ChatRequest {
    pub fn to_duck_chat_request(&mut self) {
        // Process messages to extract file sources
        let mut processed_messages: Vec<Message> = Vec::new();

        for msg in &self.messages {
            if let Some(Content::Text(text)) = &msg.content {
                let (clean_content, files) = extract_file_sources(text);

                // Add file messages first (marked as assistant)
                for (file_path, file_content) in files {
                    let file_msg = format!(
                        "============ FILE: {} ============\n{}",
                        file_path, file_content
                    );
                    processed_messages.push(
                        Message::builder()
                            .role(Role::Assistant)
                            .content(Content::Text(file_msg))
                            .build(),
                    );
                }

                // Add the clean user message (only if there's content left)
                if !clean_content.is_empty() {
                    processed_messages.push(
                        Message::builder()
                            .role(msg.role.clone().unwrap_or(Role::User))
                            .content(Content::Text(clean_content))
                            .build(),
                    );
                }
            } else {
                // Non-text content, keep as-is
                processed_messages.push(msg.clone());
            }
        }

        self.messages = processed_messages;

        
    }

    pub fn compress_messages(&mut self) {
        if self.messages.len() > 1 || self.compressed {
            self.messages = vec![
                Message::builder()
                    .role(Role::User)
                    .content(Content::Text(compress_messages(&self.messages)))
                    .build(),
            ];
            self.compressed = true;
        }
    }
}

// ==================== Duck APi Response Body ====================
#[derive(Deserialize)]
pub struct DuckChatCompletion {
    pub message: Option<String>,
    pub created: u64,
    #[serde(default = "default_id")]
    pub id: String,
    pub model: Option<String>,
}

fn default_id() -> String {
    "chatcmpl-123".to_owned()
}

// ==================== Response Body ====================
#[derive(Serialize, TypedBuilder)]
pub struct ChatCompletion<'a> {
    #[builder(default, setter(into))]
    #[serde(default = "default_id")]
    id: Option<String>,

    object: &'static str,

    #[builder(default, setter(into))]
    created: Option<u64>,

    model: &'a str,

    choices: Vec<Choice>,

    #[builder(default, setter(into))]
    usage: Option<Usage>,
}

#[derive(Serialize, TypedBuilder)]
pub struct Choice {
    index: usize,

    #[builder(default, setter(into))]
    #[serde(skip_serializing_if = "Option::is_none")]
    message: Option<Message>,

    #[builder(default, setter(into))]
    #[serde(skip_serializing_if = "Option::is_none")]
    delta: Option<Message>,

    #[builder(setter(into))]
    logprobs: Option<String>,

    #[builder(setter(into))]
    finish_reason: Option<&'static str>,
}

#[derive(Serialize, TypedBuilder)]
pub struct Usage {
    prompt_tokens: i32,
    completion_tokens: i32,
    total_tokens: i32,
}
