(function ($) {

var Man = Backbone.Model.extend({
    initialize: function(){
        alert("Hey, you create me!");

        this.bind("change:name", function(){
            var name = this.get('name');
            alert("You changed my name as: " + name);
        });

        this.bind("error", function(model, error){
            alert(error);
        });

    },


    defaults: {
        name: "Joe",
        age: '34'
    },

    validate: function(attributes){
        if (attributes.name == ''){
            return "name could not be blank!";
        }
    },

    about: function(){
        return "My name is " + this.get('name') +", I am " + this.get('age') + " years old.";
    }
});

var man = new Man;
man.set({nationality: "China", gender: "Male"});
alert("name: " + man.get('name') +"\nage: " + man.get('age') + "\nnationality: " + man.get('nationality') + "\ngender: " + man.get('gender'));
alert(man.about());

man.set({name: "Jack"});
//alert("name: " + man.get('name') +"\nage: " + man.get('age') + "\nnationality: " + man.get('nationality') + "\ngender: " + man.get('gender'));
//alert(man.about());

man.set({name: ''});
man.save();

})(jQuery);