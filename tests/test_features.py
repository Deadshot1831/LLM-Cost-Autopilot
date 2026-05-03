from autopilot.features import PromptFeatures, extract_features


class TestExtractFeatures:
    def test_returns_promptfeatures(self):
        f = extract_features("Hello world.")
        assert isinstance(f, PromptFeatures)

    def test_token_count_is_word_based(self):
        f = extract_features("one two three four")
        assert f.token_count == 4

    def test_instruction_verbs_detected(self):
        f = extract_features("Analyze the following data and compare results.")
        assert f.instruction_verb_count >= 2

    def test_no_instruction_verbs(self):
        f = extract_features("hello there friend")
        assert f.instruction_verb_count == 0

    def test_constraint_count(self):
        f = extract_features("List exactly 3 items, no more than 50 words each, in JSON.")
        assert f.constraint_count >= 2

    def test_has_context_when_long_quoted_block(self):
        prompt = 'Summarize: """' + ("text " * 80) + '"""'
        f = extract_features(prompt)
        assert f.has_context is True

    def test_no_context_short_prompt(self):
        f = extract_features("Translate hello to French.")
        assert f.has_context is False

    def test_output_format_complexity_simple(self):
        f = extract_features("What is 2 + 2?")
        assert f.output_format_complexity == 0

    def test_output_format_complexity_structured(self):
        f = extract_features("Return a JSON object with keys 'name' and 'age'.")
        assert f.output_format_complexity >= 1

    def test_to_vector_returns_numeric_list(self):
        f = extract_features("Analyze the data.")
        vec = f.to_vector()
        assert isinstance(vec, list)
        assert all(isinstance(x, (int, float)) for x in vec)
        assert len(vec) == 5
