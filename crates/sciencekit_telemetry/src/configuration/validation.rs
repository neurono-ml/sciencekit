//! Validation helpers for telemetry endpoints, headers and resource fields
//! (spec `telemetry-export`).

use std::collections::BTreeMap;
use std::time::Duration;

use sciencekit_common::SKError;
use sciencekit_common::builders::sk_validate_hyperparameter;

pub(crate) fn sk_validate_service_name(service_name: Option<&str>) -> Result<String, SKError> {
    let trimmed = service_name.map(str::trim).unwrap_or("");
    sk_validate_hyperparameter(
        "service_name",
        !trimmed.is_empty(),
        "service name must be provided and non-empty",
    )?;
    Ok(trimmed.to_string())
}

pub(crate) fn sk_validate_optional_text(
    field_name: &'static str,
    value: Option<String>,
) -> Result<Option<String>, SKError> {
    match value {
        None => Ok(None),
        Some(raw) => {
            let trimmed = raw.trim();
            sk_validate_hyperparameter(
                field_name,
                !trimmed.is_empty(),
                format!("{field_name} must not be empty when set"),
            )?;
            Ok(Some(trimmed.to_string()))
        }
    }
}

pub(crate) fn sk_validate_resource_attributes(
    attributes: &BTreeMap<String, String>,
) -> Result<(), SKError> {
    let all_keys_non_empty = attributes.keys().all(|key| !key.trim().is_empty());
    sk_validate_hyperparameter(
        "resource_attribute",
        all_keys_non_empty,
        "resource attribute keys must be non-empty",
    )
}

pub(crate) fn sk_validate_endpoint(endpoint: &str) -> Result<(), SKError> {
    if endpoint
        .chars()
        .any(|character| character.is_whitespace() || character.is_control())
    {
        return Err(SKError::InvalidHyperparameter {
            name: "endpoint",
            reason: format!("endpoint `{endpoint}` must not contain whitespace or control bytes"),
        });
    }

    let remainder = endpoint
        .strip_prefix("http://")
        .or_else(|| endpoint.strip_prefix("https://"));
    let Some(after_scheme) = remainder else {
        return Err(SKError::InvalidHyperparameter {
            name: "endpoint",
            reason: format!("endpoint `{endpoint}` must start with `http://` or `https://`"),
        });
    };

    let authority = after_scheme
        .split(['/', '?', '#'])
        .next()
        .unwrap_or_default();
    sk_validate_authority(endpoint, authority)
}

fn sk_validate_authority(endpoint: &str, authority: &str) -> Result<(), SKError> {
    if authority.is_empty() {
        return Err(SKError::InvalidHyperparameter {
            name: "endpoint",
            reason: format!("endpoint `{endpoint}` must include a host"),
        });
    }

    let (host, optional_port, is_bracketed_ipv6) = if let Some(after_bracket) =
        authority.strip_prefix('[')
    {
        let Some((ipv6_host, tail)) = after_bracket.split_once(']') else {
            return Err(SKError::InvalidHyperparameter {
                name: "endpoint",
                reason: format!("endpoint `{endpoint}` has an unclosed IPv6 host bracket"),
            });
        };
        let port = if tail.is_empty() {
            None
        } else if let Some(port_text) = tail.strip_prefix(':') {
            Some(port_text)
        } else {
            return Err(SKError::InvalidHyperparameter {
                name: "endpoint",
                reason: format!("endpoint `{endpoint}` has malformed characters after IPv6 host"),
            });
        };
        (ipv6_host, port, true)
    } else if let Some((host_part, port_part)) = authority.rsplit_once(':') {
        (host_part, Some(port_part), false)
    } else {
        (authority, None, false)
    };

    sk_validate_hyperparameter(
        "endpoint",
        !host.is_empty() && (is_bracketed_ipv6 || !host.contains(':')),
        format!("endpoint `{endpoint}` must include a valid host"),
    )?;

    if let Some(port_text) = optional_port {
        let parsed_port = port_text
            .parse::<u16>()
            .ok()
            .filter(|&port| port >= 1)
            .ok_or_else(|| SKError::InvalidHyperparameter {
                name: "endpoint",
                reason: format!(
                    "endpoint `{endpoint}` has an invalid port `{port_text}`; expected 1..=65535"
                ),
            })?;
        let _ = parsed_port;
    }

    Ok(())
}

pub(crate) fn sk_validate_headers(headers: &BTreeMap<String, String>) -> Result<(), SKError> {
    for (name, value) in headers {
        let valid_name = !name.is_empty()
            && name.bytes().all(|byte| {
                byte.is_ascii_alphanumeric()
                    || matches!(
                        byte,
                        b'!' | b'#'
                            | b'$'
                            | b'%'
                            | b'&'
                            | b'\''
                            | b'*'
                            | b'+'
                            | b'-'
                            | b'.'
                            | b'^'
                            | b'_'
                            | b'`'
                            | b'|'
                            | b'~'
                    )
            });
        sk_validate_hyperparameter(
            "headers",
            valid_name,
            format!("header name `{name}` is not a valid ASCII token"),
        )?;

        let valid_value = !value
            .bytes()
            .any(|byte| byte == b'\r' || byte == b'\n' || byte == 0);
        sk_validate_hyperparameter(
            "headers",
            valid_value,
            format!("header `{name}` contains forbidden control bytes"),
        )?;
    }
    Ok(())
}

pub(crate) fn sk_validate_positive_duration(
    field_name: &'static str,
    duration: Duration,
) -> Result<(), SKError> {
    sk_validate_hyperparameter(
        field_name,
        !duration.is_zero(),
        format!("{field_name} must be strictly positive"),
    )
}
