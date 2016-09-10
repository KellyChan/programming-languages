      
   var draw_data = function(data){
     var svg = d3.select("svg");
     var circle = svg.selectAll("circle").data(data);

       
    circle.enter().append("circle")
      .attr("cx", function(d,i){return 40*(i+1);})
      .attr("cy", function(d,i){return 30*(i+1);})
      .attr("r", 0);
       
    circle.transition()
      .duration(750)
      .attr("r", function(d) { return Math.sqrt(d); })
      .attr("cx", function(d,i){return 40*(i+1);})
      .attr("cy", function(d,i){return 30*(i+1);})
      .attr("r", function(d) { return Math.sqrt(d*0.3); });
    
    circle.exit()
        .transition()
        .duration(750)
        .attr("r", 0)
        .remove();
   }     ;
     

     setTimeout(function(){draw_data([1000,2000,3000]);},2000) ;
     setTimeout(function(){draw_data([100,200,300,400]);},4000); 
     setTimeout(function(){draw_data([500,700,600]);},6000) ;
     setTimeout(function(){draw_data([1200,1000]);},8000) ;
