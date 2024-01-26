//
// Copyright 2022 Battelle Memorial Institute
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice, this list of conditions
// and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions
// and the following disclaimer in the documentation and/or other materials provided with the distribution.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED
// WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
// INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE
// GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
// WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//

#ifndef INDEXTREE_H
#define INDEXTREE_H

#include <unordered_set>
#include <list>
#include <algorithm>
#include <map>
#include <set>

#include "comet/Dialect/IndexTree/Transforms/Tensor.h"

#include "mlir/IR/Location.h"

class Index_Tree;

class TreeNode
{
public:
  int index = -1;
  UnitExpression *expr = nullptr;
  std::vector<TreeNode *> children;
  TreeNode *parent = nullptr;
  IterDomain *inputDomain = nullptr;
  IterDomain *outputDomain = nullptr;
  int id = 0;

public:
  TreeNode() {}

  TreeNode(int index)
  {
    this->index = index;
    this->parent = nullptr;
  }

  TreeNode(int index, TreeNode *parent)
  {
    this->index = index;
    this->parent = parent;
  }

  TreeNode(UnitExpression *e, TreeNode *parent)
  {
    this->expr = e;
    this->parent = parent;
  }

  void appendChild(TreeNode *node)
  { // const T& node
    for (auto &child : children)
    {
      assert(child != node && "child already exists!");
    }
    this->children.push_back(node);
  }

  void prependChild(TreeNode *node)
  { // const T& node
    for (auto &child : children)
    {
      assert(child != node && "child already exists!");
    }
    this->children.insert(children.begin(), node);
  }

  int getIndex()
  {
    return this->index;
  }

  void setIndex(int index)
  {
    this->index = index;
  }

  TreeNode *getParent()
  {
    return this->parent;
  }

  void setParent(TreeNode *parent)
  {
    this->parent = parent;
  }

  UnitExpression *getExpression()
  {
    return this->expr;
  }

  TreeNode *getChild(int i)
  {
    return this->children.at(i);
  }

  vector<TreeNode *> &getChildren()
  {
    return children;
  }

  void setChildren(std::vector<TreeNode *> &children)
  {
    this->children.clear();
    for (auto &c : children)
    {
      c->setParent(this);
      this->children.push_back(c);
    }
  }

  IterDomain *getInputDomain()
  {
    return inputDomain;
  }

  void setInputDomain(IterDomain *d)
  {
    inputDomain = d;
  }

  IterDomain *getOutputDomain()
  {
    return outputDomain;
  }

  void setOutputDomain(IterDomain *d)
  {
    outputDomain = d;
  }

  string getName();

  bool isRealIndexNode()
  {
    return expr == nullptr && index != -1;
  }

  bool isFillerIndexNode()
  {
    return expr == nullptr && index == -1;
  }

  bool isComputeNode()
  {
    return expr != nullptr;
  }

  void moveChildrenToNewParent(TreeNode *host)
  {
    for (auto &c : children)
    {
      c->setParent(host);
      host->appendChild(c);
    }
    children.clear();
  }

  int getId() const;

  void setId(int Id);

  string str()
  {
    return getName();
  }

  void print(int depth);
};

class Index_Tree
{
  vector<unique_ptr<TreeNode>> nodes;
  vector<unique_ptr<UnitExpression>> expressions;
  std::map<UnitExpression *, TreeNode *> exprToNode;
  std::map<void *, unique_ptr<Tensor>> valueToTensor;
  std::map<void *, int> indexLabelToId;
  unsigned int indexID = 0;

public:
  IndicesType getIndices(std::vector<mlir::Value> &lbls);
  Tensor *getOrCreateTensor(mlir::Value v, std::vector<mlir::Value> &allIndexLabels, FormatsType &formats);

  vector<TreeNode *> getNodes();

  TreeNode *registerNode(unique_ptr<TreeNode> &node)
  {
    auto nodePtr = node.get();
    nodePtr->setId(nodes.size());
    nodes.push_back(std::move(node));
    return nodePtr;
  }

  TreeNode *addIndexNode(int index, TreeNode *parent)
  {
    auto node = make_unique<TreeNode>(index, parent);
    auto nodePtr = registerNode(node);
    parent->appendChild(nodePtr);
    return nodePtr;
  }

  TreeNode *addIndexNode(int index, TreeNode *parent, IterDomain *domain)
  {
    TreeNode *node = addIndexNode(index, parent);
    node->setInputDomain(domain);
    return node;
  }

  TreeNode *addRootNode()
  {
    assert(nodes.empty() && "tree is not empty!");
    auto node = make_unique<TreeNode>(-1);
    auto nodePtr = registerNode(node);
    return nodePtr;
  }

  TreeNode *addComputeNode(unique_ptr<UnitExpression> expr, TreeNode *parent);

  TreeNode *getNodeForExpression(UnitExpression *e)
  {
    return exprToNode.at(e);
  }

  TreeNode *getRoot()
  {
    return nodes.at(0).get();
  }

  vector<UnitExpression *> getExpressions()
  {
    vector<UnitExpression *> ptrs;
    for (auto &e : expressions)
    {
      ptrs.push_back(e.get());
    }
    return ptrs;
  }

  vector<UnitExpression *> getExpressionsInTopoOrder();

  vector<TreeNode *> getNodesInReverseTopoOrder();

  vector<UnitExpression *> getContainingExpressions(TreeNode *node);

  vector<mlir::Operation *> getContainingTAOps();

  vector<TreeNode *> getAllIndexChildren(vector<TreeNode *> parents)
  {
    vector<TreeNode *> children;
    for (auto p : parents)
    {
      for (auto c : p->getChildren())
      {
        assert(std::find(children.begin(), children.end(), c) == children.end());
        if (c->isRealIndexNode())
          children.push_back(c);
      }
    }
    return children;
  }

  void print(string msg = "");

  static string getIndexName(int i)
  {
    return "i" + std::to_string(i);
  }

  static unique_ptr<Index_Tree> createTreeWithRoot();
};

#endif // INDEXTREE_H
