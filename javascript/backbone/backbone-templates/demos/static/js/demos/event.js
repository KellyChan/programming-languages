(function($){

    var ListView = Backbone.View.extend({

        el: $('body'),

        events: {
            'click button#add': 'addItem'
        },

        initialize: function(){
            _.bindAll(this, 'render', 'addItem');
            this.counter = 0;
            this.render();
        },

        render: function(){
            $(this.el).append("<button id='add'>Add</button>");
            $(this.el).append("<ul></ul>");
        },

        addItem: function(){
            this.counter++;
            $('ul', this.el).append("<li>counter: " + this.counter + "</li>");
        }
    });

    var listView = new ListView();

})(jQuery);


