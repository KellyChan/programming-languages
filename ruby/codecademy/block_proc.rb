def greeter
    yield
end

phrase = Proc.new do
    puts "Hello there!"
end

greeter(&phrase)
