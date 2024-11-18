#let prompt(content, lang:"md") = {
  box(
    inset: 15pt,
    width: auto,
    fill: rgb(247, 246, 243, 50%),
  )[#text(content, font: "Cascadia Mono", size: 0.8em)]
}

#let _returns = {
  text("Returns:", font:"Cascadia Mono", weight: 600, size:9pt)
}

#let _params = {
  text("Parameters:", font:"Cascadia Mono", weight: 600, size:9pt)
}

#let _details = {
  text("Details:", font:"Cascadia Mono", weight: 600, size:9pt)
}

#let space = "  "