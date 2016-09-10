package com.todo.controller;

import org.springframework.stereotype.Controller;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.ResponseBody;

@Controller
public class HelloController {
  @RequestMapping("hi")
  @ResponseBody
  public String hi() {
    return "Hello world.";
  }
}

