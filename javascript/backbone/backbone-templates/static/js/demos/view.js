(function($){

    var ListView = Backbone.View.extend({
        el: $('body'),

        initialize: function(){
            _.bindAll(this, 'render');
            this.render();
        },

        render: function(){
            $(this.el).append("<ul><li>Hello World!</li></ul>");
        }
    });

    var listView = new ListView();

})(jQuery);


