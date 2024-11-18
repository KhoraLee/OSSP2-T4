#let team_name = "T04"
#let title = "강화학습을 이용한 주식 트레이딩 시스템 개발"
#let sub_title = "오픈소스 SW프로젝트2"
#let authors = (
  team_name,
  "202011339 이승윤",
  "202211328 윤찬규",
  "202011375 조익성"
)

#let head = {
  [
    #text(weight: 700)[#team_name]
    #text(weight: 400)[#sub_title]
    #h(1fr)
    #text(weight: 400)[#title]
    
    #line(length: 100%, stroke: 0.2pt)
  ]
}

#let project(title: "", authors: (), logo: none, body) = {
  set text(9pt, font: "Pretendard")
  set heading(numbering: "1.")
  set page(columns: 1, numbering: "1  /  1", number-align: center, header: head, margin: 5em)
  show outline.entry.where(level: 1): it => {
    v(25pt, weak:true)
    strong(it)
  }
  show heading : it => { it; v(0.5em);}
  
  align(center)[
    #block(text(weight: 800, 1.75em, title))
  ]
  pad(
    top: 0.5em,
    bottom: 0.5em,
    x: 2em,
    grid(
      columns: (1fr,) * calc.min(1, authors.len()),
      gutter: 1em,
      ..authors.map(author => align(center, author)),
    ),
  )
  set par(justify: true)
  outline(title: "목 차", depth: 5, indent: 1em, fill: repeat(text(weight: 700)[.#h(0.5em)]))

  body
}