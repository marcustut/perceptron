#import "template.typ": *
#import "@preview/cetz:0.1.2": canvas, plot, palette, draw

#let legend-item(point, name, style) = {
  draw.content(
    (point.at(0)+2, point.at(1)), (point.at(0)+2, point.at(1)), 
    frame: "rect",
    padding: .3em,
    fill: style.fill,
    stroke: none,
    [],
  )
  draw.content(
    (point.at(0)+2.4, point.at(1)+0.1), (point.at(0)+5.5, point.at(1)),
    [
      #v(.2em)
      #text(name, size: .7em, weight: "bold")
    ]
  )
}

#show: project.with(
  title: "Multi Layer Perceptron",
  authors: (
    (name: "Lee Kai Yang (23205838)", email: "kai.y.lee@ucdconnect.ie"),
  ),
  date: "December 4, 2023",
  font: "CMU Serif",
  monofont: "CMU Typewriter Text",
)

= Introduction

The aim of this paper is to implement a multi layer perceptron (aka Neural Network) without using any external libraries.

== Language used

The language I chose is C++, specifically #strong("C++20"). 

== How to compile 

To compile the project you will need the following tools installed:

= Math

== Feedforward

== Backpropagation

To produce better results, a network needs to adjust its weights and biases "smartly" or also known as training, hence backpropagation is introduced to the network where after the network propagates until the output nodes, we calculate the loss function and propagate the network backwards layer by layer to adjust the weights and biases for each layer.