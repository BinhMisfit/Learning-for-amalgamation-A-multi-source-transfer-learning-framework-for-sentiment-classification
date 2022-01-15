import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertPreTrainedModel
from transformers import RobertaModel, BertPreTrainedModel

# 1
class BertLSTMAttn(nn.Module):
    def __init__(self, config_phobert, config_bert, attention_type='dot', hidden_size=128, dropout_prob=0.1, maxlen=128, bert_only=False, phobert_only=False):
        super(BertLSTMAttn, self).__init__()

# 2
#class BertLSTMAttn(BertPreTrainedModel):
#    def __init__(self, config_bert, attention_type='dot', hidden_size=128, dropout_prob=0.1):
#        super().__init__(config_bert)

        self.config_phobert = config_phobert
        self.config_bert = config_bert
        self.maxlen = maxlen
        self.hidden_size = hidden_size
        self.dropout_prob = dropout_prob
        self.attention_type = attention_type
        self.num_labels = self.config_bert.num_labels
        self.bert_only=bert_only
        self.phobert_only=phobert_only

        self.phobert = RobertaModel(config_phobert)
        self.bert = BertModel(config_bert)

        self.gate = nn.Linear(2*768, 2)
        self.gate_bn = nn.BatchNorm1d(self.maxlen)

        self.lstm = nn.LSTM(config_bert.hidden_size, self.hidden_size, bidirectional=True, batch_first=True)
        self.linear = nn.Linear(2*self.hidden_size, 2*self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_prob)
        self.classifier = nn.Linear(6*self.hidden_size, config_bert.num_labels)

        # 1
        self.phobert.init_weights()
        self.bert.init_weights()


        # 2
        #self.init_weights()


    '''
    def init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()
    '''


    #def from_pretrained(self, model_phobert, model_bert):
    #    #self.phobert.from_pretrained(model_phobert, config=self.config_phobert)
    #    self.bert.from_pretrained(model_bert, config=self.config_bert)

    def attention(self, lstm, final_hidden_state):
        if self.attention_type == 'dot':
            attention_weights = torch.bmm(lstm, final_hidden_state.unsqueeze(2)).squeeze(2)
        elif self.attention_type == 'general':
            attention_weights = torch.bmm(self.linear(lstm), final_hidden_state.unsqueeze(2)).squeeze(2)
        attention_weights = F.softmax(attention_weights, 1)

        attention = torch.bmm(lstm.transpose(1, 2), attention_weights.unsqueeze(2)).squeeze(2)
        return attention

    def forward(
        self,
        input_ids_phobert=None,
        input_ids_bert=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
    ):
        # bert embeddings
        outputs_phobert = self.phobert(
            input_ids_phobert,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        ) #torch.Size([bs, seqlen, 768])

        outputs_bert = self.bert(
            input_ids_bert,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        ) #torch.Size([bs, seqlen, 768])

        # MixtureOfExperts
        if self.bert_only is True:
            sequence_output = outputs_bert[0]
        elif self.phobert_only is True:
            sequence_output = outputs_phobert[0]
        else:
            method = 'SpMixtureOfExperts' #MixtureOfExperts, SpMixtureOfExperts, SpAttnMixtureOfExperts
            if method == 'MixtureOfExperts':
                inputs_gate = torch.cat([outputs_phobert[0], outputs_bert[0]], dim=2) #torch.Size([bs, seqlen, 768*2])

                inputs_gate = inputs_gate.detach()
                inputs_gate = self.gate_bn(inputs_gate)
                outputs_gate = self.gate(inputs_gate.float())
                outputs_gate_softmax = F.softmax(outputs_gate, dim=-1) ##torch.Size([bs, seqlen, 2])

                sequence_output = torch.stack([outputs_phobert[0], outputs_bert[0]], dim=-2) # bs x seqlen x #experts x output_size
                sequence_output = torch.sum(outputs_gate_softmax.unsqueeze(-1) * sequence_output, dim=-2) # bs x seqlen x output_size

            elif method == 'SpMixtureOfExperts':
                inputs_gate = torch.cat([outputs_phobert[0], outputs_bert[0]], dim=2) #torch.Size([bs, seqlen, 768*2])

                inputs_gate = inputs_gate.detach()
                outputs_gate = self.gate(inputs_gate.float())
                outputs_gate = self.gate_bn(outputs_gate)
                outputs_gate_softmax = F.sigmoid(outputs_gate)

                sequence_output = torch.stack([outputs_phobert[0], outputs_bert[0]], dim=-2) # bs x seqlen x #experts x output_size
                sequence_output = torch.sum(outputs_gate_softmax.unsqueeze(-1) * sequence_output, dim=-2) # bs x seqlen x output_size

            elif method == 'SpAttnMixtureOfExperts':
                pass


        # final layers
        lstm, (hn, _) = self.lstm(sequence_output)
        final_hn_layer = hn.view(self.lstm.num_layers, self.lstm.bidirectional+1, hn.shape[1], self.hidden_size)[-1, :, :, :]
        final_hidden_state = torch.cat([final_hn_layer[i, :, :] for i in range(final_hn_layer.shape[0])], dim=1)
        attention = self.attention(lstm, final_hidden_state)
        avg_pool = torch.mean(lstm, 1)
        max_pool, _ = torch.max(lstm, 1)
        cat = torch.cat((avg_pool, max_pool, attention), 1)

        cat = self.dropout(cat)
        logits = self.classifier(cat)

        #outputs = (logits,) + outputs[2:]
        outputs = (logits,)

        if labels is not None:
            if self.num_labels == 1:
                loss_fct = nn.MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs



# 1
class BertGRUAttn(nn.Module):
    def __init__(self, config_phobert, config_bert, attention_type='dot', hidden_size=128, dropout_prob=0.1, maxlen=128):
        super(BertGRUAttn, self).__init__()

# 2
#class BertLSTMAttn(BertPreTrainedModel):
#    def __init__(self, config_bert, attention_type='dot', hidden_size=128, dropout_prob=0.1):
#        super().__init__(config_bert)

        self.config_phobert = config_phobert
        self.config_bert = config_bert
        self.maxlen = maxlen
        self.hidden_size = hidden_size
        self.dropout_prob = dropout_prob
        self.attention_type = attention_type
        self.num_labels = self.config_bert.num_labels

        self.phobert = RobertaModel(config_phobert)
        self.bert = BertModel(config_bert)

        self.gate = nn.Linear(2*768, 2)
        self.gate_bn = nn.BatchNorm1d(self.maxlen)

        #self.lstm = nn.LSTM(config_bert.hidden_size, self.hidden_size, bidirectional=True, batch_first=True)
        self.lstm = nn.GRU(config_bert.hidden_size, self.hidden_size, bidirectional=True, batch_first=True)
        self.linear = nn.Linear(2*self.hidden_size, 2*self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_prob)
        self.classifier = nn.Linear(6*self.hidden_size, config_bert.num_labels)

        # 1
        self.phobert.init_weights()
        self.bert.init_weights()


        # 2
        #self.init_weights()


    '''
    def init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()
    '''


    #def from_pretrained(self, model_phobert, model_bert):
    #    #self.phobert.from_pretrained(model_phobert, config=self.config_phobert)
    #    self.bert.from_pretrained(model_bert, config=self.config_bert)

    def attention(self, lstm, final_hidden_state):
        if self.attention_type == 'dot':
            attention_weights = torch.bmm(lstm, final_hidden_state.unsqueeze(2)).squeeze(2)
        elif self.attention_type == 'general':
            attention_weights = torch.bmm(self.linear(lstm), final_hidden_state.unsqueeze(2)).squeeze(2)
        attention_weights = F.softmax(attention_weights, 1)

        attention = torch.bmm(lstm.transpose(1, 2), attention_weights.unsqueeze(2)).squeeze(2)
        return attention

    def forward(
        self,
        input_ids_phobert=None,
        input_ids_bert=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
    ):
        # bert embeddings
        outputs_phobert = self.phobert(
            input_ids_phobert,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        ) #torch.Size([bs, seqlen, 768])

        outputs_bert = self.bert(
            input_ids_bert,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        ) #torch.Size([bs, seqlen, 768])

        # MixtureOfExperts
        method = 'SpMixtureOfExperts' #MixtureOfExperts, SpMixtureOfExperts, SpAttnMixtureOfExperts
        if method == 'MixtureOfExperts':
            inputs_gate = torch.cat([outputs_phobert[0], outputs_bert[0]], dim=2) #torch.Size([bs, seqlen, 768*2])

            inputs_gate = inputs_gate.detach()
            inputs_gate = self.gate_bn(inputs_gate)
            outputs_gate = self.gate(inputs_gate.float())
            outputs_gate_softmax = F.softmax(outputs_gate, dim=-1) ##torch.Size([bs, seqlen, 2])

            sequence_output = torch.stack([outputs_phobert[0], outputs_bert[0]], dim=-2) # bs x seqlen x #experts x output_size
            sequence_output = torch.sum(outputs_gate_softmax.unsqueeze(-1) * sequence_output, dim=-2) # bs x seqlen x output_size

        elif method == 'SpMixtureOfExperts':
            inputs_gate = torch.cat([outputs_phobert[0], outputs_bert[0]], dim=2) #torch.Size([bs, seqlen, 768*2])

            inputs_gate = inputs_gate.detach()
            outputs_gate = self.gate(inputs_gate.float())
            outputs_gate = self.gate_bn(outputs_gate)
            outputs_gate_softmax = F.sigmoid(outputs_gate)

            sequence_output = torch.stack([outputs_phobert[0], outputs_bert[0]], dim=-2) # bs x seqlen x #experts x output_size
            sequence_output = torch.sum(outputs_gate_softmax.unsqueeze(-1) * sequence_output, dim=-2) # bs x seqlen x output_size

        elif method == 'SpAttnMixtureOfExperts':
            pass


        # final layers
        lstm, (hn, _) = self.lstm(sequence_output)
        final_hn_layer = hn.view(self.lstm.num_layers, self.lstm.bidirectional+1, hn.shape[1], self.hidden_size)[-1, :, :, :]
        final_hidden_state = torch.cat([final_hn_layer[i, :, :] for i in range(final_hn_layer.shape[0])], dim=1)
        attention = self.attention(lstm, final_hidden_state)
        avg_pool = torch.mean(lstm, 1)
        max_pool, _ = torch.max(lstm, 1)
        cat = torch.cat((avg_pool, max_pool, attention), 1)

        cat = self.dropout(cat)
        logits = self.classifier(cat)

        #outputs = (logits,) + outputs[2:]
        outputs = (logits,)

        if labels is not None:
            if self.num_labels == 1:
                loss_fct = nn.MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs




'''
# 1
class BertGRUAttn(nn.Module):
    def __init__(self, config_phobert, config_bert, attention_type='dot', hidden_size=128, dropout_prob=0.1, maxlen=128):
        super(BertGRUAttn, self).__init__()

        self.config_phobert = config_phobert
        self.config_bert = config_bert
        self.maxlen = maxlen
        self.hidden_size = hidden_size
        self.dropout_prob = dropout_prob
        self.attention_type = attention_type
        self.num_labels = self.config_bert.num_labels

        self.phobert = RobertaModel(config_phobert)
        self.bert = BertModel(config_bert)

        self.gate = nn.Linear(2*768, 2)
        self.gate_bn = nn.BatchNorm1d(self.maxlen)

        self.gru  = nn.GRU(config_phobert.hidden_size, self.hidden_size, bidirectional=True, batch_first=True)
        self.linear = nn.Linear(2*self.hidden_size, 2*self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_prob)
        self.classifier = nn.Linear(6*self.hidden_size, config_bert.num_labels)

        # 1
        self.phobert.init_weights()
        self.bert.init_weights()


        # 2
        #self.init_weights()

    #def from_pretrained(self, model_phobert, model_bert):
    #    #self.phobert.from_pretrained(model_phobert, config=self.config_phobert)
    #    self.bert.from_pretrained(model_bert, config=self.config_bert)

    def attention(self, gru, final_hidden_state):
        if self.attention_type == 'dot':
            attention_weights = torch.bmm(gru, final_hidden_state.unsqueeze(2)).squeeze(2)
        elif self.attention_type == 'general':
            attention_weights = torch.bmm(self.linear(gru), final_hidden_state.unsqueeze(2)).squeeze(2)
        attention_weights = F.softmax(attention_weights, 1)

        attention = torch.bmm(gru.transpose(1, 2), attention_weights.unsqueeze(2)).squeeze(2)
        return attention

    def forward(
        self,
        input_ids_phobert=None,
        input_ids_bert=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
    ):
        # bert embeddings
        outputs_phobert = self.phobert(
            input_ids_phobert,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        ) #torch.Size([bs, seqlen, 768])
        print('outputs_phobert', outputs_phobert[0].shape)

        outputs_bert = self.bert(
            input_ids_bert,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        ) #torch.Size([bs, seqlen, 768])
        print('outputs_bert', outputs_bert[0].shape)

        # MixtureOfExperts
        method = 'SpMixtureOfExperts' #MixtureOfExperts, SpMixtureOfExperts, SpAttnMixtureOfExperts
        if method == 'MixtureOfExperts':
            inputs_gate = torch.cat([outputs_phobert[0], outputs_bert[0]], dim=2) #torch.Size([bs, seqlen, 768*2])

            inputs_gate = inputs_gate.detach()
            inputs_gate = self.gate_bn(inputs_gate)
            outputs_gate = self.gate(inputs_gate.float())
            outputs_gate_softmax = F.softmax(outputs_gate, dim=-1) ##torch.Size([bs, seqlen, 2])

            sequence_output = torch.stack([outputs_phobert[0], outputs_bert[0]], dim=-2) # bs x seqlen x #experts x output_size
            sequence_output = torch.sum(outputs_gate_softmax.unsqueeze(-1) * sequence_output, dim=-2) # bs x seqlen x output_size

        elif method == 'SpMixtureOfExperts':
            inputs_gate = torch.cat([outputs_phobert[0], outputs_bert[0]], dim=2) #torch.Size([bs, seqlen, 768*2])

            inputs_gate = inputs_gate.detach()
            outputs_gate = self.gate(inputs_gate.float())
            outputs_gate = self.gate_bn(outputs_gate)
            outputs_gate_softmax = F.sigmoid(outputs_gate)

            sequence_output = torch.stack([outputs_phobert[0], outputs_bert[0]], dim=-2) # bs x seqlen x #experts x output_size
            sequence_output = torch.sum(outputs_gate_softmax.unsqueeze(-1) * sequence_output, dim=-2) # bs x seqlen x output_size

        elif method == 'SpAttnMixtureOfExperts':
            pass

        print('sequence_output', sequence_output.shape)

        # final layers
        gru, (hn, _) = self.gru(sequence_output)
        print('gru', gru.shape)
        print('hn', hn.shape)
        final_hn_layer = hn.view(self.lstm.num_layers, self.lstm.bidirectional+1, hn.shape[1], self.hidden_size)[-1, :, :, :]
        final_hn_layer = hn.view(self.gru.num_layers, self.gru.bidirectional+1, hn.shape[1], self.hidden_size)[-1, :, :, :]
        final_hidden_state = torch.cat([final_hn_layer[i, :, :] for i in range(final_hn_layer.shape[0])], dim=1)
        attention = self.attention(gru, final_hidden_state)
        avg_pool = torch.mean(gru, 1)
        max_pool, _ = torch.max(gru, 1)
        cat = torch.cat((avg_pool, max_pool, attention), 1)

        cat = self.dropout(cat)
        logits = self.classifier(cat)

        #outputs = (logits,) + outputs[2:]
        outputs = (logits,)

        if labels is not None:
            if self.num_labels == 1:
                loss_fct = nn.MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs
'''
