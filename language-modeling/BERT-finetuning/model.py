# Copyright 2021 Sony Group Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import nnabla.functions as F
import nnabla.parametric_functions as PF


def bert_embed(input_ids, token_type_ids=None, position_ids=None, vocab_size=30522, embed_dim=768,
               num_pos_ids=512, dropout_prob=0.1, test=True):
    """Construct the embeddings from word, position and token type."""

    batch_size = input_ids.shape[0]
    seq_len = input_ids.shape[1]
    if position_ids is None:
        position_ids = F.arange(0, seq_len)
        position_ids = F.broadcast(F.reshape(
            position_ids, (1,)+position_ids.shape), (batch_size,) + position_ids.shape)
    if token_type_ids is None:
        token_type_ids = F.constant(val=0, shape=(batch_size, seq_len))

    embeddings = PF.embed(input_ids, vocab_size,
                          embed_dim, name='word_embeddings')
    position_embeddings = PF.embed(
        position_ids, num_pos_ids, embed_dim, name='position_embeddings')
    token_type_embeddings = PF.embed(
        token_type_ids, 2, embed_dim, name='token_type_embeddings')

    embeddings += position_embeddings
    embeddings += token_type_embeddings
    embeddings = PF.layer_normalization(
        embeddings, batch_axis=(0, 1), eps=1e-12, name='embed')

    if dropout_prob > 0.0 and not test:
        embeddings = F.dropout(embeddings, dropout_prob)

    return embeddings


def bert_layer(hs, num_layers=12, embed_dim=768, num_heads=12,
               dim_feedforward=3072, activation=None, attention_mask=None,
               head_mask=None, encoder_hidden_states=None,
               encoder_attention_mask=None, dropout_prob=0.1, test=True):
    """ Generate Transformer Layers"""
    # Transpose the input to the shape (L,B,E) accepted by parameter function transformer_encode
    hs = F.transpose(hs, (1, 0, 2))
    for i in range(num_layers):
        if test:
            hs = PF.transformer_encode(hs, embed_dim, num_heads, dim_feedforward=dim_feedforward,
                                       dropout=0.0, activation=activation,
                                       name='encoder{:02d}'.format(i))
        else:
            hs = PF.transformer_encode(hs, embed_dim, num_heads, dim_feedforward=dim_feedforward,
                                       dropout=dropout_prob, activation=activation,
                                       name='encoder{:02d}'.format(i))
    # Transpose back to (B,L,E)
    self_outputs = F.transpose(hs, (1, 0, 2))

    return self_outputs


def bert_encode(hs, attention_mask=None, head_mask=None, num_attention_layers=12,
                num_attention_embed_dim=768, num_attention_heads=12,
                num_attention_dim_feedforward=3072, attention_activation=None,
                encoder_hidden_states=None, encoder_attention_mask=None,
                dropout_prob=0.1, test=True):
    layer_outputs = bert_layer(hs, num_layers=num_attention_layers,
                               embed_dim=num_attention_embed_dim,
                               num_heads=num_attention_heads,
                               dim_feedforward=num_attention_dim_feedforward,
                               activation=attention_activation,
                               attention_mask=attention_mask, head_mask=head_mask,
                               encoder_hidden_states=encoder_hidden_states,
                               encoder_attention_mask=encoder_attention_mask,
                               dropout_prob=dropout_prob, test=test)

    return layer_outputs


def bert_pool(hs, out_dim=768):
    '''
    BERT Pooler, Pool the model by taking hidden state corresponding to the first token
    hs: Hidden State (B, L, E)
    '''

    first_token_tensor = hs[:, 0]
    pooled_output = F.tanh(
        PF.affine(first_token_tensor, out_dim, name="pooler"))

    return pooled_output


class BertModel():
    def __init__(self):
        pass

    def __call__(self, args, input_ids, attention_mask=None, token_type_ids=None,
                 position_ids=None,
                 head_mask=None, vocab_size=30522, num_embed_dim=768, num_pos_ids=512,
                 num_attention_layers=12, num_attention_embed_dim=768, num_attention_heads=12,
                 num_attention_dim_feedforward=3072, attention_activation=None, pool_outmap=768,
                 embed_dropout_prob=0.1, attention_dropout_prob=0.1,
                 encoder_hidden_states=None, encoder_attention_mask=None, test=True):
        input_shape = input_ids.shape

        if attention_mask is None:
            attention_mask = F.constant(val=1, shape=(128, 128))
        if token_type_ids is None:
            token_type_ids = F.constant(val=0, shape=input_shape)

        if len(attention_mask.shape) == 3:
            extended_attention_mask = attention_mask[:, None, :, :]
        elif len(attention_mask.shape) == 2:
            extended_attention_mask = attention_mask[:, None, None, :]
        else:
            raise ValueError("Wrong shape for input_ids (shape {}) or attention_mask (shape {})".format(
                input_shape, attention_mask.shape))

        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        encoder_extended_attention_mask = None
        head_mask = None

        embedding_output = bert_embed(input_ids, position_ids=position_ids,
                                      token_type_ids=token_type_ids, vocab_size=vocab_size,
                                      embed_dim=num_embed_dim, num_pos_ids=num_pos_ids,
                                      dropout_prob=embed_dropout_prob, test=test)

        encoder_outputs = bert_encode(embedding_output,
                                      attention_mask=attention_mask,
                                      head_mask=head_mask,
                                      num_attention_layers=num_attention_layers,
                                      num_attention_embed_dim=num_attention_embed_dim,
                                      num_attention_heads=num_attention_heads,
                                      num_attention_dim_feedforward=num_attention_dim_feedforward,
                                      attention_activation=attention_activation,
                                      encoder_hidden_states=encoder_hidden_states,
                                      encoder_attention_mask=encoder_extended_attention_mask,
                                      dropout_prob=args.attention_dropout, test=test)

        pooled_output = bert_pool(encoder_outputs, out_dim=pool_outmap)

        return pooled_output


class BertForSequenceClassification():
    def __init__(self):
        self.bert = BertModel()

    def __call__(self, args, input_ids, attention_mask=None, token_type_ids=None,
                 position_ids=None, head_mask=None, labels=None, num_labels=2,
                 vocab_size=30522, num_embed_dim=768, num_pos_ids=512,
                 num_attention_layers=12, num_attention_embed_dim=768,
                 num_attention_heads=12, num_attention_dim_feedforward=3072,
                 attention_activation=None, pool_outmap=768, embed_dropout_prob=0.1,
                 attention_dropout_prob=0.1, dropout_prob=0.1,
                 test=True):

        pooled_output = self.bert(args, input_ids,
                                  attention_mask=attention_mask,
                                  token_type_ids=token_type_ids,
                                  position_ids=position_ids,
                                  head_mask=head_mask,
                                  vocab_size=vocab_size,
                                  num_embed_dim=num_embed_dim,
                                  num_pos_ids=num_pos_ids,
                                  num_attention_layers=num_attention_layers,
                                  num_attention_embed_dim=num_attention_embed_dim,
                                  num_attention_heads=num_attention_heads,
                                  num_attention_dim_feedforward=num_attention_dim_feedforward,
                                  attention_activation=attention_activation,
                                  pool_outmap=pool_outmap,
                                  embed_dropout_prob=embed_dropout_prob,
                                  attention_dropout_prob=attention_dropout_prob,
                                  test=test)

        if not test:
            pooled_output = F.dropout(pooled_output, p=dropout_prob)
        logits = PF.affine(pooled_output, num_labels,
                           base_axis=1, name='affine_seq_class')

        label = F.reshape(labels, (-1, 1), inplace=False)
        if args.task_name == "sts-b":
            loss = F.mean((logits-label)**2)
        else:
            loss = F.mean(F.softmax_cross_entropy(logits, label))
        error = F.sum(F.top_n_error(logits, label))

        return loss, logits, error
