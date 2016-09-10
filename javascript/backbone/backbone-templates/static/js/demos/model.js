(function($){
    World = Backbone.Model.extend({
        name: null
    });

    Worlds = Backbone.Collection.extend({
        initialize: function(models, options){
            this.bind("add", options.view.addOneWorld);
        }
    });

    AppView = Backbone.View.extend({

        el: $("body"),

        initialize: function (){
            this.worlds = new Worlds(null, {view: this})
        },

        events: {
            "click #check": "checkIn",
        },

        checkIn: function(){
            var world_name = prompt("Where are you from?");
            if (world_name == "") world_name = "unknown";
            var world = new World({ name: world_name });
            this.worlds.add(world);
        },

        addOneWorld: function(model) {
            $("#world-list").append("<li>Welcome to <b>" + model.get('name') + "</b>!");
        }
    });

    var appview = new AppView;

})(jQuery);