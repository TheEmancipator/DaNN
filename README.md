# DaNN
An implementation of a discourse-aware tree-structured neural network <BR>

This implementation is for the original research paper: [Kangwook Lee, Sanggyu Han and Sung-Hyon Myaeng, "A discourse-aware neural network-based text model for document-level text classification", Journal of Information Science, Accepted, DOI: 10.1177/0123456789123456] <BR>


This implementation is based on the Theano implementation of the tree LSTM by Ofir Nachum (https://github.com/ofirnachum/tree_rnn) <BR>

To analyze documents' discourse, we use the DPLP parser [Ji and Eisenstein, ACL 2014] (https://github.com/jiyfeng/DPLP) <BR>

To train the EDU embeddings, we use Paragraph Vector [Le and Mikolov], implemented in the Gensim library (https://radimrehurek.com/gensim) <BR>
