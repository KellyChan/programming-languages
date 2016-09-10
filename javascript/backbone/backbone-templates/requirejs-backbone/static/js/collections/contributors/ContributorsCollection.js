define(['underscore', 'backbone',
        'models/contributor/ContributorModel'
], function(_, Backbone, ContributorModel){

    var ContributorsCollection = Backbone.Collection.extend({

        model: ContributorModel,

        initialize: function(models, options){

        },

        url: function (){
            return 'https://api.github.com/repos/kellychan/backbone-templates';
        },

        parse: function(data){
            var uniqueArray = this.removeDuplicates(data.data);
            return uniqueArray;
        },

        removeDuplicates: function(myArray){

            var length = myArray.length;
            var ArrayWithUniqueValues = [];

            var objectCounter = {};

            for (i=0; i < length; i++){
                var currentMemboerOfArrayKey = JSON.stringify(myArray[i]);
                var currentMemboerOfArrayValue = myArray[i];

                if (objectCounter[currentMemboerOfArrayKey] === undefined){
                    ArrayWithUniqueValues.push(currentMemboerOfArrayValue);
                    objectCounter[currentMemboerOfArrayKey] = 1;
                } else {
                    objectCounter[currentMemboerOfArrayKey]++;
                }
            }

            return ArrayWithUniqueValues;
        }
    });

    return ContributorsCollection;
});


