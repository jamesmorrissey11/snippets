def remove_patterns(text, patterns_to_remove):
    for pattern in patterns_to_remove:
        text = text.replace(pattern, "")
    return text


def clean_columns(df):
    df.columns = [c.lower().strip().replace(" ", "_") for c in df.columns]
