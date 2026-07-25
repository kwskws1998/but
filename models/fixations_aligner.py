from tokenizeraligner.models.tokenizer_aligner import TokenizerAligner


class FixationsAligner:
    """Map token-level fixation features between different tokenizers."""

    @staticmethod
    def map_features_lcs(features, input_ids_original, text_tokenized_model):
        """Restore mapped features to the original token sequence."""
        common_tokens = TokenizerAligner().longest_common_subsequence(
            input_ids_original, text_tokenized_model
        )
        common_token_index = 0
        mapped_token_index = 0
        corrected_features = []

        for original_token_index in range(len(input_ids_original)):
            if common_token_index >= len(common_tokens):
                corrected_features.extend(
                    [0] * (len(input_ids_original) - len(corrected_features))
                )
                break

            if (
                input_ids_original[original_token_index]
                != common_tokens[common_token_index]
            ):
                corrected_features.append(0)
                continue

            while (
                mapped_token_index < len(text_tokenized_model)
                and text_tokenized_model[mapped_token_index]
                != common_tokens[common_token_index]
            ):
                mapped_token_index += 1

            if (
                mapped_token_index >= len(text_tokenized_model)
                or mapped_token_index >= len(features)
            ):
                corrected_features.extend(
                    [0] * (len(input_ids_original) - len(corrected_features))
                )
                break

            corrected_features.append(features[mapped_token_index])
            common_token_index += 1
            mapped_token_index += 1

        return corrected_features

    @staticmethod
    def map_features_between_tokens(
        features,
        tokens_id_mapped,
        text_tokenized_model,
        text_tokenized_fix,
        mode="mean",
    ):
        """Aggregate fixation features and assign them to mapped model tokens."""
        feature_index = 0
        mapped_feature_index = 0
        mapped_features = []

        for pair in tokens_id_mapped:
            fixation_pair = []
            pair_index = 0
            while (
                pair_index < len(pair[1])
                and feature_index < len(text_tokenized_fix)
            ):
                if text_tokenized_fix[feature_index] == pair[1][pair_index]:
                    fixation_pair.append(features[feature_index])
                    pair_index += 1
                feature_index += 1
            if pair_index != len(pair[1]):
                raise ValueError("Fixation-token alignment exceeded source tokens")

            pair_index = 0
            mapped_pair_indices = []
            while (
                pair_index < len(pair[0])
                and mapped_feature_index < len(text_tokenized_model)
            ):
                if (
                    text_tokenized_model[mapped_feature_index]
                    == pair[0][pair_index]
                ):
                    mapped_pair_indices.append(mapped_feature_index)
                    pair_index += 1
                mapped_feature_index += 1
            if pair_index != len(pair[0]) or not mapped_pair_indices:
                raise ValueError("Fixation-token alignment has no target tokens")

            if mode == "mean":
                feature_value = sum(fixation_pair) / len(mapped_pair_indices)
            elif mode == "max":
                feature_value = max(fixation_pair)
            elif mode == "sum":
                feature_value = sum(fixation_pair)
            else:
                raise ValueError("mode must be mean, max, or sum")

            while len(mapped_features) < mapped_feature_index:
                if text_tokenized_model[len(mapped_features)] in pair[0]:
                    mapped_features.append(feature_value)
                else:
                    mapped_features.append(0)

        return mapped_features

    @staticmethod
    def map_fixations_between_tokens_correct(
        features,
        tokens_id_mapped,
        input_ids_original,
        text_tokenized_model,
        text_tokenized_fix,
        return_all=False,
    ):
        """Map fixation features and align them to the original model tokens."""
        if not isinstance(features[0], list):
            features = [features]

        mapped_features = []
        corrected_features = []

        for batch_index, token_mapping in enumerate(tokens_id_mapped):
            if token_mapping is None:
                corrected = TokenizerAligner().adjust_list_length(
                    features[batch_index], input_ids_original[batch_index]
                )
                corrected_features.append(corrected)
                continue

            mapped = FixationsAligner.map_features_between_tokens(
                features[batch_index],
                token_mapping,
                text_tokenized_model["input_ids"][batch_index],
                text_tokenized_fix["input_ids"][batch_index],
            )
            mapped_features.append(mapped)
            corrected = FixationsAligner.map_features_lcs(
                mapped,
                input_ids_original[batch_index],
                text_tokenized_model["input_ids"][batch_index],
            )
            corrected_features.append(corrected)

        if return_all:
            return corrected_features, mapped_features
        return corrected_features
