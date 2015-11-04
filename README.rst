Earthquake Classifier
=====================

To install this package

::

    pip install git+ssh://git@github.com/ajcobo/earthquakeclassifier.git

Basic Example

::

      from earthquakeclassifier import earthquakeclassifier
      
      instance = classify.Classifier()
      
      instance.classify([
                        ['mensaje de terremoto'], 146, 174], 
                        ['mal mensaje, me quiero ir a mi cama', 122, 742],
                        # [message, followers, friends (or followees)]                      
                       ])

The result is

::

      [[0.9000455, 0.0999545],
      [0.71414738, 0.28585262]]
     # [ Prob. of being relevant, Prob. of not being relevant ]

The classify method accepts an array of arrays in the form [message,
followers, friends (or followees)]

The results are in the form [ Probability of being relevant for the
earthquake, Probability of not being relevant ]

To implement the classifier in an application, the recomendation is to
not classify very short messages and to have a threshold to accept the
message to be above 0.8 at least (please test it in practice).
