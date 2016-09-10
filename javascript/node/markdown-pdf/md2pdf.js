var md_file = "./file_name.md"
var output_file = "./outputs/file_name.pdf"

var markdownpdf = require("markdown-pdf")
var fs = require("fs")

fs.createReadStream(md_file)
  .pipe(markdownpdf())
  .pipe(fs.createWriteStream(output_file))
