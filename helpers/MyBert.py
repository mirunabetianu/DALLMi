from typing import Optional, Union, Tuple
from torch.nn import CrossEntropyLoss, MSELoss, BCEWithLogitsLoss
from transformers import BertPreTrainedModel, BertModel
from torch import nn
import torch
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.models.bert.modeling_bert import BertEmbeddings, BertEncoder, BertPooler


class MyBert(BertPreTrainedModel):

    def __init__(self, config):
        super().__init__(config)

        self.num_labels = config.num_labels
        self.config = config

        self.bert = BertModel(config)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def _forward_init(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
    ):
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape, device)

        # If a 2D ou 3D attention mask is provided for the cross-attention
        # we need to make broadcastabe to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        return input_ids, attention_mask, token_type_ids, position_ids, head_mask, inputs_embeds, \
            encoder_hidden_states, encoder_attention_mask, extended_attention_mask, encoder_extended_attention_mask

    def forward_mix_embed(self, unlabelled_input_id, unlabelled_attention_mask, positive_input_id, positive_attention_mask, mu):
        unlabelled_input_id, attention_mask1, token_type_ids, position_ids, head_mask, inputs_embeds, encoder_hidden_states, \
            encoder_attention_mask, extended_attention_mask1, encoder_extended_attention_mask = self._forward_init(
                input_ids=unlabelled_input_id, attention_mask=unlabelled_attention_mask)

        embedding_output1 = self.bert.embeddings(
            input_ids=unlabelled_input_id, position_ids=position_ids, token_type_ids=token_type_ids, inputs_embeds=inputs_embeds
        )

        positive_input_id, attention_mask2, token_type_ids, position_ids, head_mask, inputs_embeds, encoder_hidden_states, \
            encoder_attention_mask, extended_attention_mask2, encoder_extended_attention_mask = self._forward_init(
                input_ids=positive_input_id, attention_mask=positive_attention_mask)

        embedding_output2 = self.bert.embeddings(
            input_ids=positive_input_id, position_ids=position_ids, token_type_ids=token_type_ids, inputs_embeds=inputs_embeds
        )

        embedding_output = mu * embedding_output1 + (1.0 - mu) * embedding_output2

        # need to take max of both to ensure we don't miss attending to any value
        extended_attention_mask = torch.max(extended_attention_mask1, extended_attention_mask2)
        encoder_outputs = self.bert.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
        )

        sequence_output = encoder_outputs[0]
        pooled_output = self.bert.pooler(sequence_output)

        outputs = (sequence_output, pooled_output,) + encoder_outputs[
            1:
        ]

        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        return logits
    
    def forward_mix_sent(self, unlabelled_input_ids, unlabelled_attention_mask, positive_input_ids, positive_attention_mask, mu):
        unlabelled_logits = (self.forward(unlabelled_input_ids, unlabelled_attention_mask)).logits
        positive_logits = (self.forward(positive_input_ids, positive_attention_mask)).logits
        y = mu * unlabelled_logits + (1.0-mu) * positive_logits
        return y

    def forward_mix_encoder(self, unlabelled_input_ids, unlabelled_attention_mask, positive_input_ids, positive_attention_mask, mu):
        unlabelled_output = self.bert(unlabelled_input_ids, unlabelled_attention_mask)
        positive_output = self.bert(positive_input_ids, positive_attention_mask)
        unlabelled_pooled_output = self.dropout(unlabelled_output[1])
        positive_pooled_output = self.dropout(positive_output[1])
        pooled_output = mu * unlabelled_pooled_output + (1.0-mu) * positive_pooled_output
        logits = self.classifier(pooled_output)
        return logits
    
    def forward_mix_embeddings(self, unlabelled_input_ids, unlabelled_attention_mask, positive_input_ids, positive_attention_mask, mu):
        word_embeddings = self.bert.embeddings.word_embeddings
        position_embeddings = self.bert.embeddings.position_embeddings
        token_type_embeddings = self.bert.embeddings.token_type_embeddings

        unlabelled_input_id, attention_mask1, unlabelled_token_type_ids, unlabelled_position_ids, head_mask, inputs_embeds, encoder_hidden_states, \
            encoder_attention_mask, extended_attention_mask1, encoder_extended_attention_mask = self._forward_init(
                input_ids=unlabelled_input_ids, attention_mask=unlabelled_attention_mask)

        
        positive_input_id, attention_mask2, positive_token_type_ids, positive_position_ids, head_mask, inputs_embeds, encoder_hidden_states, \
            encoder_attention_mask, extended_attention_mask2, encoder_extended_attention_mask = self._forward_init(
                input_ids=positive_input_ids, attention_mask=positive_attention_mask)
        

        unlabelled_word_embeddings_output = word_embeddings(unlabelled_input_ids)
        positive_word_embeddings_output = word_embeddings(positive_input_ids)

        # unlabelled_position_embeddings_output = position_embeddings(unlabelled_position_ids)
        # positive_position_embeddings_output = position_embeddings(positive_position_ids)

        unlabelled_token_type_embeddings_output = token_type_embeddings(unlabelled_token_type_ids)
        positive_token_type_embeddings_output = token_type_embeddings(positive_token_type_ids)

        word_embeddings_output = mu * unlabelled_word_embeddings_output + (1.0 - mu) * positive_word_embeddings_output
        # position_embeddings_output = mu * unlabelled_position_embeddings_output + (1.0 - mu) * positive_position_embeddings_output
        token_type_embeddings_output = mu * unlabelled_token_type_embeddings_output + (1.0 - mu) * positive_token_type_embeddings_output

        embeddings = word_embeddings_output + token_type_embeddings_output
        # if self.position_embedding_type == "absolute":
        #     embeddings += position_embeddings_output


        extended_attention_mask = torch.max(extended_attention_mask1, extended_attention_mask2)
        encoder_outputs = self.bert.encoder(
            embeddings,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
        )

        sequence_output = encoder_outputs[0]
        pooled_output = self.bert.pooler(sequence_output)

        outputs = (sequence_output, pooled_output,) + encoder_outputs[
            1:
        ]

        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        return logits
