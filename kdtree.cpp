#include <iostream>
#include <vector>
#include <string>
#include <stack>
#include <functional>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <ctime>

#define ABS(x) ((x)>0?(x):-(x))
#define SQ(x) ((x)*(x))

using namespace std;
using namespace std::placeholders;
using it = vector<int>::iterator;

const int TESTCASE = 10000;//10~29000
const int TRAINSET_SIZE = 70000;//100~70000
const int DIM=14;
const int K=3;//3~10

const string originFileName = "";
const string normalFileName = "";
const string dosFileName = "";
const string normalNormalizedFileName = "";
const string dosNormalizedFileName = "";
const string trainFileName = "";
const string normalTestFileName = "";
const string dosTestFileName = "";
const string rocFileName = "";

//数据索引
int index[TRAINSET_SIZE];

struct Data{
    vector<float> values;

    Data(int k = DIM) : values(k) {}
    Data(vector<float>& init):values(init){
        values.reserve(DIM);
    }
    inline float& operator [](int k){
        return values[k];
    }
    inline float dist(Data& obj){
        float dis = 0;
        for (unsigned int i=0; i < values.size(); ++i){
            dis += SQ(values[i]-obj[i]);
        }
        return sqrt(dis);
    }
    inline void setValue(float val, int k){
        values[k] = val;
    }
};
ostream& operator << (ostream& os, const Data& obj){
    os << "( ";
    for (auto i : obj.values){
        os << i <<" ";
    }
    os << ")";
    return os;
}

using DataSet = vector<Data>;
//训练集、正常数据测试集、dos攻击数据测试集
DataSet trainSet(TRAINSET_SIZE), normalTestSet(TESTCASE), dosTestSet(TESTCASE);

struct Node{
    int dim=-1;
    float val=0;
    int beg=-1, last=-1;
    Node *left=nullptr, *right=nullptr;
    bool toLeft=false;
};

//根据方差返回分割所依照的维度
inline int getDim(int beg, int last){
    vector<float> diff;
    for (int j=0; j<DIM; ++j){
        float sum=0;
        for (int i=beg; i<= last; ++i){
            sum += trainSet[index[i]][j];
        }
        float mean = sum/TRAINSET_SIZE;
        float variance = 0;
        for (int i=beg; i<=last; ++i){
            variance += SQ(trainSet[index[i]][j]-mean);
        }
        diff.push_back(variance);
    }
    float maxVar = -1;
    int dim = -1;
    for (unsigned int i=0; i<diff.size(); ++i) {
        if (diff[i] > maxVar){
            maxVar = diff[i];
            dim = i;
        }
    }
    return dim;
}
inline bool cmp(int i, int j, int dim){
    return trainSet[i][dim] < trainSet[j][dim];
}
void build(Node* root, int beg, int last){
    root->beg = beg;
    root->last = last;
    if (last-beg < K)  return;
    root->dim = getDim(beg, last);
    sort(index+beg, index+last+1, bind(cmp,_1,_2,root->dim));
    int split = beg+(last-beg)/2;
    root->val = trainSet[index[split]][root->dim];
    root->left = new Node();
    root->right = new Node();
    build(root->left, beg, split);
    build(root->right, split+1, last);
}

void destroy(Node* root){
    if (!root)  return;
    destroy(root->left);
    destroy(root->right);
    delete root;
}

pair<int, float> findNearest(Node* root, Data data){
    stack<Node*> path;
    Node* nd=root;
    float dis = DIM;
    float loc = -1;
    while (nd->left || nd->right){
        path.push(nd);
        if (data[nd->dim] <= nd->val){
            nd->toLeft = true;
            nd = nd->left;
        } else{
            nd = nd->right;
        }
    }
    for (int i=nd->beg; i<=nd->last; ++i){
        if (data.dist(trainSet[index[i]]) < dis){
            loc = i;
            dis = data.dist(trainSet[index[i]]);
        }
    }

    while(!path.empty()){
        nd = path.top();
        path.pop();
        bool toLeft = nd->toLeft;
        nd->toLeft = false;
        if (ABS(data[nd->dim] - nd->val) < dis){
            nd = toLeft ? nd->right : nd->left;
            while (nd->left || nd->right){
                path.push(nd);
                if (data[nd->dim] < nd->val){
                    nd->toLeft = true;
                    nd = nd->left;
                } else{
                    nd = nd->right;
                }
            }
            for (int i=nd->beg; i<=nd->last; ++i){
                if (data.dist(trainSet[index[i]]) < dis){
                    loc = i;
                    dis = data.dist(trainSet[index[i]]);
                }
            }
        }
    }
    return pair<int, float>(loc, dis);
}

inline string format(string line){
    //1,5,6,8,9,23~31

    int before, after;
    string result;

    //1
    before = line.find_first_of(',');
    result += line.substr(0, before);

    //5
    for (int i=1; i<4; ++i){
        before = line.find_first_of(',', before+1);
    }
    after = line.find_first_of(',' ,before+1);
    result += " "+line.substr(before+1, after-before-1);

    //6
    before = after;
    after = line.find_first_of(',', after+1);
    result += " " + line.substr(before+1, after - before - 1);

    //8
    for (int i=6; i<8; ++i){
        before = line.find_first_of(',', before+1);
    }
    after = line.find_first_of(',' ,before+1);
    result += " "+line.substr(before+1, after-before-1);

    //9
    before = after;
    after = line.find_first_of(',', after+1);
    result += " " + line.substr(before+1, after - before - 1);

    //23~31
    for (int i=9; i<23; ++i){
        before = line.find_first_of(',', before+1);
    }
    after = line.find_first_of(',' ,before+1);

    for (int i=23; i<=31; ++i){
        result += " " + line.substr(before+1, after - before - 1);
        before = after;
        after = line.find_first_of(',', after+1);
    }

    return result;
}

//将原始数据集分为正常/DOS攻击数据集，存储在文件中
void dataPrepare(){
    cout << "Separating origin files..." << endl;
    ofstream normal(normalFileName, ios::binary);
    ofstream dos(dosFileName, ios::binary);
    ifstream fin(originFileName, ios::binary);
    if (!fin.is_open())
        cout << originFileName << " not open" << endl;
    if (!normal.is_open())
        cout << normalFileName << " not open" << endl;
    if (!dos.is_open())
        cout << dosFileName << " not open" << endl;

    string line;
    string result;
    int k;
    //for (int i=0; i<100; ++i){
      //  getline(fin, line);
    while (getline(fin, line)){
        k = line.find_last_of(',') + 1;
        if (line.substr(k, 2)=="no"){
            normal << format(line) << '\n';
        } else if (line[k]=='t' || line.substr(k, 2)=="ba" || line.substr(k, 2)=="la"
                   || line.substr(k, 2)=="np" || line.substr(k, 2)=="po" || line.substr(k, 2)=="sm") {
            dos << format(line) << '\n';
        }
    }
    fin.close();
    normal.close();
    dos.close();
    cout << "Origin file generated two new files." << endl;
}

//辅助函数，用于计算每一维的最大最小值
void getMaxMin(string fileName, vector<float>& maximum, vector<float>& minimum){
    ifstream fin(fileName, ios::binary);
    if (!fin.is_open()){
        cout << fileName << " not open" << endl;
    }

    string line;
    float val;
    while (getline(fin, line)){
        istringstream rec(line);
        for (int i=0; i<DIM; ++i){
            rec >> val;
            if (val > maximum[i]){
                maximum[i] = val;
            } else if (val < minimum[i]){
                minimum[i] = val;
            }
        }
    }
    fin.close();
}

//辅助函数，数据标准化、归一化，并导出数据到文件中
void normalized(string sourceFileName, string targetFileName, vector<float>& minimum, vector<float>& base){
    ifstream fin(sourceFileName, ios::binary);
    if (!fin.is_open()){
        cout << sourceFileName << " not open" << endl;
    }
    ofstream nor(targetFileName, ios::binary);
    if (!nor.is_open()){
        cout << targetFileName << " is not open" << endl;
    }
    string line;
    float val;
    while (getline(fin, line)){
        string newLine;
        istringstream rec(line);
        for (int i=0; i<DIM; ++i){
            rec >> val;
            val = (val - minimum[i])/base[i];
            ostringstream os;
            os << val;
            newLine += " " + os.str();
        }
        nor << newLine << '\n';
    }

    fin.close();
    nor.close();
}

//将数据集标准、归一化
void normalize(){
    cout << "Normalizing files..." << endl;
    vector<float> minimum;
    vector<float> maximum;
    vector<float> base;
    string line;
    float val;
    {
        ifstream fin(normalFileName, ios::binary);
        getline(fin, line);
        istringstream rec(line);
        for (int i=0; i<DIM; ++i){
            rec >> val;
            maximum.push_back(val);
            minimum.push_back(val);
        }
        fin.close();
    }
    getMaxMin(normalFileName, maximum, minimum);
    getMaxMin(dosFileName, maximum, minimum);
    for (int i=0; i<DIM; ++i){
        base.push_back(maximum[i]-minimum[i]);
        cout << base[i] << endl;
    }
    normalized(normalFileName, normalNormalizedFileName, minimum, base);
    normalized(dosFileName, dosNormalizedFileName, minimum, base);
    cout << "Normalizing finished." << endl;
}

//选取部分数据作为训练集、测试集存储在文件中
void prepareTrainTest(){
    cout << "Preparing train set..." << endl;
    //pick 70000 normal data as train set
    ifstream fin(normalNormalizedFileName, ios::binary);
    if (!fin.is_open()){
        cout << normalNormalizedFileName << " not open" << endl;
    }
    ofstream train(trainFileName, ios::binary);
    if (!train.is_open()){
        cout << trainFileName << " not open" << endl;
    }
    string line;
    for (int i=0; i<TRAINSET_SIZE; ++i){
        getline(fin, line);
        train << line << '\n';
    }
    train.close();
    cout << "Train set prepared." << endl;

    cout << "Preparing test set..." << endl;
    //pick 10000 normal data as test set
    ofstream test(normalTestFileName, ios::binary);
    if (!test.is_open()){
        cout << normalTestFileName << " not open" << endl;
    }
    for (int i=0; i<TESTCASE; ++i){
        getline(fin, line);
        test << line << '\n';
    }
    test.close();
    fin.close();

    //pick 10000 dos data as test set
    fin.open(dosNormalizedFileName, ios::binary);
    test.open(dosTestFileName, ios::binary);
    for (int i=0; i<TESTCASE; ++i){
        getline(fin, line);
        test << line << '\n';
    }
    test.close();
    fin.close();
    cout << "Test set prepared." << endl;
}

//从文件数据中创建内存数据集
int createDataSet(int n, string fileName, DataSet& dataSet){
    cout << "Create data set from " << fileName << endl << endl;

    ifstream fin(fileName, ios::binary);
    if (!fin.is_open()){
        cout << fileName << "not open during create test set" << endl;
    }
    string line;
    float val;
    for (int i=0; i<n; ++i){
        Data newData;
        getline(fin, line);
        istringstream rec(line);
        for (int j=0; j<DIM; ++j){
            rec >> val;
            newData[j] = val;
        }
        dataSet[i] = newData;
    }
    fin.close();

    cout << "End with " << fileName << endl << endl;
    return n;
}

//测试
void doTest(Node* root, float lowerBound, float delta, int times){
    //output the test result
    ofstream fout(rocFileName, ios::app);
    if (!fout.is_open()){
        cout << rocFileName << " not open." << endl;
    }
    fout << "**Train set size = " << TRAINSET_SIZE << ", test cases = " << TESTCASE << ".\n";
    fout << "**Threshold set from " << lowerBound << " to " << lowerBound + delta * (times-1) << " by " << delta << ".\n";

    cout << "Start test...\n";
    //do the test
    vector<float> threshold(times, lowerBound);
    vector<int> hitNormal(times, 0);
    vector<int> hitDos(times, 0);
    float dis;
    for (int i = 1; i < times; ++i){
        threshold[i] += delta * i;
    }

    for (int i = 0; i < TESTCASE; ++i){
        dis = findNearest(root, normalTestSet[i]).second;
        for (int j = 0; j < times; ++j){
            if (dis < threshold[j]){
                ++hitNormal[j];
            }
        }
    }
    for (int i = 0; i < TESTCASE; ++i){
        dis = findNearest(root, dosTestSet[i]).second;
        for (int j = 0; j < times; ++j){
            if (dis > threshold[j]){
                ++hitDos[j];
            }
        }
    }
    for (int i=0; i<times; ++i){
        fout << "Threshold = " << threshold[i] << ", "
            << (double)hitNormal[i]/TESTCASE << ", " << (double)hitDos[i]/TESTCASE << '\n';
    }
    fout << "*****************************************************\n";
    fout.close();
    cout << "Test finished.\n";
}

int main()
{
    /*
    //准备数据文件，只需执行1次
    dataPrepare();
    normalize();
    prepareTrainTest();
    */

    //创建KD tree
    createDataSet(TRAINSET_SIZE, trainFileName, trainSet);
    for (int i=0; i < TRAINSET_SIZE; ++i){
        index[i] = i;
    }
    Node* root = new Node();
    cout << "Start building KD tree..." << endl;
    build(root, 0, TRAINSET_SIZE-1);
    cout << "KD tree built." << endl;

    //创建内存测试数据集
    createDataSet(TESTCASE, normalTestFileName, normalTestSet);
    createDataSet(TESTCASE, dosTestFileName, dosTestSet);

    //执行测试，结果输出到文件中
    doTest(root, 0.0001, 0.0003, 8);

    //释放KD tree的内存空间
    destroy(root);

    return 0;
}
