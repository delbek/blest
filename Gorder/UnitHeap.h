/*
MIT License

Copyright (c) 2016, Hao Wei.

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*/

#ifndef _UNITHEAP_H
#define _UNITHEAP_H

#include<cstdlib>
#include<vector>
#include<climits>
#include<algorithm>

#include "Util.h"

namespace Gorder
{

using namespace std;

const long long int INITIALVALUE=0;


class ListElement{
public:
	long long int key;
	long long int prev;
	long long int next;
};


class HeadEnd{
public:
	long long int first;
	long long int second;

	HeadEnd(){
		first=second=-1;
	}
};

	
class UnitHeap{
public:
	long long int* update;
	ListElement* LinkedList;
	vector<HeadEnd> Header;
	long long int top;
	long long int heapsize;

	UnitHeap(long long int size);
	~UnitHeap();
	void DeleteElement(const long long int index);
	long long int ExtractMax();
	void DecrementKey(const long long int index);
	void DecreaseTop();
	void ReConstruct();
	
	void IncrementKey(const long long int index){
		long long int key=LinkedList[index].key;
		const long long int head=Header[key].first;
		const long long int prev=LinkedList[index].prev;
		const long long int next=LinkedList[index].next;

		if(head!=index){
			LinkedList[prev].next=next;
			if(next>=0)
				LinkedList[next].prev=prev;
			
			long long int headprev=LinkedList[head].prev;
			LinkedList[index].prev=headprev;
			LinkedList[index].next=head;
			LinkedList[head].prev=index;
			if(headprev>=0)
				LinkedList[headprev].next=index;
		}

		LinkedList[index].key++;
#ifndef Release
		if(key+1>=Header.size()){
			cout << "IncrementKey: key+1>=Header.size()\t" << key+1 << endl;
			quit();
		}
#endif
		if(Header[key].first==Header[key].second)
			Header[key].first=Header[key].second=-1;
		else if(Header[key].first==index)
			Header[key].first=next;
		else if(Header[key].second==index)
			Header[key].second=prev;
		
		key++;
		Header[key].second=index;
		if(Header[key].first<0)
			Header[key].first=index;

		if(LinkedList[top].key<key){
			top=index;
		}

		if(key+4>=Header.size()){
			Header.resize(Header.size()*1.5);
		}		
	}
};

}


#endif
