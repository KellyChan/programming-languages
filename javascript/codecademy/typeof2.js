var languages = {
    english: "Hello!",
    french: "Bonjour!",
    notALanguage: 4,
    spanish: "Hola!"
};

// print hello in the 3 different languages
for (var x in languages){
    if (typeof languages[x] == "string"){
        console.log(languages[x]);
    };
};
