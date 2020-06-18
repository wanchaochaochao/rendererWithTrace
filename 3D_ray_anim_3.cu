#include<iostream>
#include "book.h"
#include "cuda.h"
#include "gpu_anim.h"
#include "lodepng.h"
#include <stdio.h>
#include <stdlib.h>
#include<string>

#define INF 2e10f
#define rnd( x )( x*rand() / RAND_MAX )
/*#define SPHERE 4
#define LIGHT 2
#define GROUND 2
#define PLANE 13
#define IMAGE 4*/
#define WIDTH 1080
#define HEIGHT 1080
#define MAXDEPTH 2
#define DIM_W 1920
#define DIM_H 1080
#define MAX_LINE 64

using namespace std;

struct vec{
    float x, y, z;
    __host__ __device__ vec(){}
    __host__ __device__ vec(const float &m_x, const float &m_y, const float &m_z):x(m_x), y(m_y), z(m_z){}
    __host__ __device__ vec operator +(const vec &other)const{
        vec result = vec(this->x + other.x, this->y + other.y, this->z + other.z);return result;}
    __host__ __device__ vec operator -(const vec &other)const{
        vec result = vec(this->x - other.x, this->y - other.y, this->z - other.z);return result;}
    __host__ __device__ const vec operator -(){
        this->x = -this->x, this->y = -this->y, this->z = -this->z;return *this;}
    __host__ __device__ vec operator *(const float &timeVal)const{
        vec result = vec(this->x * timeVal, this->y * timeVal, this->z * timeVal); return result;}
    __host__ __device__ vec operator ^(const vec &other){
        this->x *= other.x, this->y *= other.y, this->z *= other.z; return *this;}
    __host__ __device__ void operator +=(const vec &other){x += other.x, y += other.y, z += other.z;}
    __host__ __device__ void operator -=(const vec &other){x -= other.x, y -= other.y, z -= other.z;}
    __host__ __device__ float norm()const{return sqrtf(x*x + y*y + z*z);}
    __host__ __device__ vec normalize()const{float normVal=this->norm();
        vec result = vec(x/normVal,y/normVal,z/normVal);return result;}
    __host__ __device__ float dot(const vec &other)const{return (x*other.x + y*other.y + z*other.z);}
    __host__ __device__ float cos(const vec &other)const{
        return this->dot(other) / (this->norm() * other.norm());}
};

struct Sphere{
    vec center, color;int num;
    float radius, reflRate = 1, refractionRate = 1.5;//光线的反射率和折射率
    Sphere(){}
    void setSphere(const vec &m_center, const float &m_radius, const vec &m_color){
        center=m_center, radius=m_radius, color=m_color;}
    //光线碰撞函数
    __device__ float hit( const vec &startPoint, const vec &rayDir, float *n ){
        vec a = center - startPoint;
        float norm_a = a.norm(), cos_a_b = rayDir.cos(a), sin_a_b = sqrtf(1-cos_a_b*cos_a_b);
        float d = norm_a * sin_a_b;
        if(cos_a_b >= 0 && d <= radius){
            *n = sqrtf(radius*radius-d*d) / radius;
            float Distance = norm_a*cos_a_b - sqrtf(radius*radius - d*d);
            return Distance;
        }
        return INF;
    }
    //光线反射函数
    __device__ void refl(const vec &startPoint, vec *rayDir){
        vec c = startPoint - this->center;
        float norm_c = c.norm(), dot_b_c_devide_norm_c = rayDir->dot(c) / (norm_c * norm_c);
        *rayDir -= c * (2*dot_b_c_devide_norm_c);
    }
    //光线折射函数,直接计算两次折射之后的光线
    __device__ void refraction(vec *startPoint, vec *rayDir){
        //光线第一次折射
        vec c = (*startPoint - center).normalize();// * (1 / radius);//计算折射光线的对称轴向量
        float  dot_b_c_with_norm_b = rayDir->dot(c);
        vec h = *rayDir - c * dot_b_c_with_norm_b;
        float cos_b_c = dot_b_c_with_norm_b / rayDir->norm();
        float sin_b_c = sqrtf(abs(1 - cos_b_c * cos_b_c));
        *rayDir = c * cos_b_c + h * (sin_b_c / refractionRate);
        //光线第一次折射之后的碰撞
        float distance = 2 * radius * sqrtf(1 - pow(sin_b_c/refractionRate,2));//计算光线碰撞前的距离
        *startPoint += *rayDir * (distance / rayDir->norm());
        //光线第二次折射
        c = (*startPoint - center).normalize();// * (1 / radius);//计算折射光线的对称轴向量
        dot_b_c_with_norm_b = rayDir->dot(c);
        h = *rayDir - c * dot_b_c_with_norm_b;
        cos_b_c = dot_b_c_with_norm_b / rayDir->norm();
        sin_b_c = sqrtf(abs(1 - cos_b_c * cos_b_c));
        *rayDir = c * cos_b_c + h * (sin_b_c * refractionRate);
    }
};

struct Ground{
    float z = 0, reflRate = 1;//光线反射率
    int num;
    vec color;
    Ground(){}
    void setGround(const float &m_z, const vec &m_color, float m_reflRate = 1){
        z = m_z, color = m_color, reflRate = m_reflRate;}
    //光线碰撞函数
    __device__ float hit(const vec &startPoint, const vec &rayDir, float *n){
        float norm_b = rayDir.norm();
        float distance = ((z - startPoint.z) / rayDir.z) * norm_b;
        if(distance >= 0){
            *n = abs(rayDir.z / norm_b);
            return distance;
        }
        return INF;
    }
    //光线反射函数
    __device__ void refl(vec *rayDir){
        rayDir->z = -rayDir->z;
    }
};

struct Image{
    float *imageData, *cpu_imageData;int width, height;int num;
    Image(){}
    ~Image(){cudaFree(imageData);}
    void setImage(const char *fileName){
        unsigned error, w, h;
        unsigned char *imageRead;
        error = lodepng_decode32_file(&imageRead, &w, &h, fileName);
        if(error) printf("error %u: %s\n", error, lodepng_error_text(error));
        width = w, height = h;
        cpu_imageData = (float *)malloc(sizeof(float) * w * h * 4);
        HANDLE_ERROR( cudaMalloc((void**)&imageData, sizeof(float) * w * h * 4) );
        //复制转换图像数据
        for(int i = 0; i < w * h * 4; i++){
            cpu_imageData[i] = (float)(imageRead[i]) / 255;}
        HANDLE_ERROR( cudaMemcpy( imageData, cpu_imageData, sizeof(float)*w*h*4, cudaMemcpyHostToDevice ) );
        free(imageRead);
        free(cpu_imageData);
    }
};

struct Plane{//平行于坐标轴的矩形平面
    vec point1, point2, color;//由矩形对角两个点来确定平面
    float reflRate = 1;int indexImage = -1;int num;
    Plane(){}
    void setPlane(const vec &m_point1, const vec &m_point2, const vec &m_color, float m_reflRate=1, int m_indexImage=-1){
        point1 = m_point1, point2 = m_point2, color = m_color, reflRate = m_reflRate, indexImage = m_indexImage;}
    __device__ float hit(const vec &startPoint, const vec &rayDir, float *n){
        float norm_b = rayDir.norm(), distance, eps = 1e-5f;
        if(point1.z == point2.z){
            *n = abs(rayDir.z / norm_b);distance = ((point1.z - startPoint.z) / rayDir.z) * norm_b;}
        else if(point1.y == point2.y){
            *n = abs(rayDir.y / norm_b);distance = ((point1.y - startPoint.y) / rayDir.y) * norm_b;}
        else if(point1.x == point2.x){
            *n = abs(rayDir.x / norm_b);distance = ((point1.x - startPoint.x) / rayDir.x) * norm_b;}
        if(distance >= 0){
            vec hitPoint = startPoint + rayDir * (distance / norm_b);
            if(hitPoint.x>=point1.x-eps && hitPoint.x<=point2.x+eps && hitPoint.y>=point1.y-eps && hitPoint.y<=point2.y+eps && hitPoint.z>=point1.z-eps && hitPoint.z<=point2.z+eps){return distance;}}
        return INF;
    }
    //光线反射函数
    __device__ void refl(vec *rayDir){
        if(point1.z == point2.z){rayDir->z = -rayDir->z;}
        if(point1.y == point2.y){rayDir->y = -rayDir->y;}
        if(point1.x == point2.x){rayDir->x = -rayDir->x;}
    }
    //纹理贴图
    __device__ void colorImage(const vec &rayDir, const vec &hitPoint, vec *color, Image *image){
        int x0, y0, w = image[indexImage].width, h = image[indexImage].height;
        if(indexImage != -1){
            if(point1.z == point2.z){
                x0 = (hitPoint.x - point1.x)*w/(point2.x-point1.x);
                y0 = (hitPoint.y - point1.y)*h/(point2.y-point1.y);}
            else if(point1.y == point2.y){
                if(rayDir.y>=point1.y){
                x0 = (point2.x - hitPoint.x)*w/(point2.x-point1.x);}
                else{
                x0 = (hitPoint.x - point1.x)*w/(point2.x-point1.x);}
                y0 = (point2.z - hitPoint.z)*h/(point2.z-point1.z);}
            else if(point1.x == point2.x){
                x0 = (point2.y - hitPoint.y)*w/(point2.y-point1.y);
                y0 = (point2.z - hitPoint.z)*h/(point2.z-point1.z);}
            color->x *= image[indexImage].imageData[(x0 + y0*w)*4];
            color->y *= image[indexImage].imageData[(x0 + y0*w)*4+1];
            color->z *= image[indexImage].imageData[(x0 + y0*w)*4+2];
        }
    }
};

struct Light{//单点光源
    vec center, color;int num;
    Light(){}
    void setLight(const vec &m_center, const vec &m_color){center = m_center, color = m_color;}
};

struct Photo{
    float x = 0, y = 0, z = 0;int num;
    vec center, photoDir;
    float focalDistance = 0.2;//焦距
    float focalScope = 0.3;//横向的广度
    float width = 960, height = 960;
    Photo(){}
    void setPhoto(const vec &m_center, const float &m_focalDistance, const float &m_focalScope, const float &m_width, const float &m_height, const vec &m_photoDir ){
        center=m_center, focalDistance=m_focalDistance, focalScope=m_focalScope, width=m_width, height=m_height, photoDir=m_photoDir;}
};

    //光线碰撞扫描函数
__device__ void hitAllObject(Light *light, Sphere *sphere, Ground *ground, Plane *plane, vec *startPoint, vec *rayDir, vec *color, Image *image, int depth, float reflRate=1 ){
    float distance, minDistance = INF, colorScale = 0;
    int indexHitSphere = -1, indexHitGround = -1, indexHitPlane = -1;
    bool isShadow = false;
    //圆球
    for(int i = 0; i < sphere->num; i++){
        distance = sphere[i].hit( *startPoint, *rayDir, &colorScale );
        if(distance < minDistance){
            minDistance = distance;
            indexHitSphere = i;
        }
    }
    //地面
    for(int i = 0; i < ground->num; i++){
        distance = ground[i].hit( *startPoint, *rayDir, &colorScale );
        if(distance < minDistance){
            minDistance = distance*(1 - 1e-5f);
            indexHitGround = i;
        }
    }
    //矩形平面
    for(int i = 0; i < plane->num; i++){
        distance = plane[i].hit( *startPoint, *rayDir, &colorScale );
        if(distance < minDistance){
            minDistance = distance*(1 - 1e-5f);
            indexHitPlane = i;
        }
    }
    //光线碰撞
    if(minDistance < INF){
        float dir_norm_minDistance = minDistance / rayDir->norm();
        *startPoint += (*rayDir) * dir_norm_minDistance;
        //计算光源的漫反射
        vec hitLightDir;
    //printf("%d ",light->num);
        for(int i = 0; i < light->num; i++){//循环遍历所有光源
            isShadow = false;
            hitLightDir = light[i].center - (*startPoint);
            float norm_hitLightDir = hitLightDir.norm();
            for(int j = 0; j < sphere->num; j++){
                distance = sphere[j].hit( *startPoint, hitLightDir, &colorScale );
                if(distance < norm_hitLightDir){isShadow = true;}
            }
            for(int j = 0; j < plane->num; j++){
                distance = plane[j].hit( *startPoint, hitLightDir, &colorScale );
                if(distance < norm_hitLightDir){isShadow = true;}
            }
            //for(int j = 0; j < 1; j++){
            //    distance = ground[j].hit( xHit, yHit, zHit, dir_xHit, dir_yHit, dir_zHit, &n );
            //}
            if(!isShadow){
                vec color0 = vec(0.8,0,0);
                colorScale = 0;//重新初始化为0,否则会有颜色残留,显示出一圈白色细线
                if(indexHitPlane != -1){
                    distance = plane[indexHitPlane].hit( light[i].center, -hitLightDir, &colorScale );
                    color0 = plane[indexHitPlane].color * reflRate;
                    plane[indexHitPlane].colorImage(*rayDir, *startPoint, &color0, image);
}
                else if(indexHitGround != -1){
                    distance = ground[indexHitGround].hit( light[i].center, -hitLightDir, &colorScale );
                    color0 = ground[indexHitGround].color * reflRate;}
                else if(indexHitSphere != -1){
                    distance = sphere[indexHitSphere].hit( light[i].center, -hitLightDir, &colorScale );
                    color0 = sphere[indexHitSphere].color * reflRate;}
                *color += ((color0 * colorScale) ^ light[i].color);
            }
        }
        //计算光线反射之后的方向向量
        if(indexHitPlane != -1){
            plane[indexHitPlane].refl(rayDir);
            reflRate *= plane[indexHitPlane].reflRate;//更新递归光线的反射率
        }
        else if(indexHitGround != -1){
            ground[indexHitGround].refl(rayDir);
            reflRate *= ground[indexHitGround].reflRate;//更新递归光线的反射率
        }
        else if(indexHitSphere != -1){
            if(indexHitSphere == 3 || indexHitSphere == 1){
                sphere[indexHitSphere].refraction(startPoint, rayDir);
            }
            else{
                sphere[indexHitSphere].refl(*startPoint, rayDir);
            }
            reflRate *= sphere[indexHitSphere].reflRate;//更新递归光线的反射率
        }
        //如果光线碰撞,递归计算下一层光线,直到最大层数
        if(depth < MAXDEPTH){
            depth += 1;
            hitAllObject( light, sphere, ground, plane, startPoint, rayDir, color, image, depth, reflRate );
        }
    }
}

__global__ void kernel_3D( uchar4 *ptr, Sphere *sphere, Ground * ground, Plane *plane, Photo *photo, Light *light, Image *image ){
    //将threadIdx/BlockIdx映射到像素位置
    float x = threadIdx.x + blockIdx.x * blockDim.x;
    float y= threadIdx.y + blockIdx.y * blockDim.y;
    int offset = x + y * blockDim.x * gridDim.x;
    //胶片上像素相对于中心的位置
    float w = (x - photo->width/2) * photo->focalScope / photo->width;
    float h = (y - photo->height/2) * photo->focalScope / photo->width;
    //光线的起点
    vec startPoint = vec(photo->center), rayDir, color = vec(0, 0, 0);
    //计算每个像素所对应的光线的向量,光线起点为相机坐标
    vec temp = vec( photo->photoDir ) * photo->focalDistance;
    float x1_y1 = sqrtf(temp.x * temp.x + temp.y * temp.y), x1_y1_z1 = temp.norm();
    rayDir.z = temp.z + h * x1_y1 / x1_y1_z1;
    rayDir.x = (x1_y1 - h * temp.z / x1_y1_z1) * temp.x / x1_y1 + w * temp.y / x1_y1;
    rayDir.y = (x1_y1 - h * temp.z / x1_y1_z1) * temp.y / x1_y1 - w * temp.x / x1_y1;
    int depth = 0;//递归计算的最大层数
    hitAllObject( light, sphere, ground, plane, &startPoint, &rayDir, &color, image, depth );
    ptr[offset].x = (unsigned char)(color.x * 255);//像素值
    ptr[offset].y = (unsigned char)(color.y * 255);
    ptr[offset].z = (unsigned char)(color.z * 255);
    ptr[offset].w = 255;
/*    ptr[offset].x = (unsigned char)(image[((int)x*3+(960-(int)y)*1620*2)*4]);//像素值
    ptr[offset].y = (unsigned char)(image[((int)x*3+(960-(int)y)*1620*2)*4+1]);
    ptr[offset].z = (unsigned char)(image[((int)x*3+(960-(int)y)*1620*2)*4+2]);
    ptr[offset].w = 255;*/
}

struct DataBlock{
    Sphere *sphere, *cpu_sphere;
    Ground *ground, *cpu_ground;
    Photo *photo, *cpu_photo;
    Light *light, *cpu_light;
    Plane *plane, *cpu_plane;
    Image *image, *cpu_image;
    bool isLockCursorInWindow = false;
};

void generate_frame( uchar4 *pixels, void *dataBlock, int ticks ) {
    dim3    grids(DIM_W/16,DIM_H/16);
    dim3    threads(16,16);
    kernel_3D<<<grids,threads>>>( pixels, ((DataBlock *)dataBlock)->sphere, ((DataBlock *)dataBlock)->ground, ((DataBlock *)dataBlock)->plane, ((DataBlock *)dataBlock)->photo, ((DataBlock *)dataBlock)->light, ((DataBlock *)dataBlock)->image );
}

void point(void *dataBlock, int x, int y){
    if(((DataBlock *)dataBlock)->isLockCursorInWindow){
        if((x-DIM_W/2) != 0 || (y-DIM_H/2) != 0){
            float dx = -(DIM_W/2 - (float)x)/4;
            float dy = -((float)y - DIM_H/2)/4;
            Photo *photo = ((DataBlock *)dataBlock)->cpu_photo;
            float w = dx * photo->focalScope / photo->width;
            float h = dy * photo->focalScope / photo->width;
            //计算每个像素所对应的光线的向量,光线起点为相机坐标
            float x1 = photo->photoDir.x * photo->focalDistance;
            float y1 = photo->photoDir.y * photo->focalDistance;
            float z1 = photo->photoDir.z * photo->focalDistance;
            float x1_y1 = sqrtf(x1 * x1 + y1 * y1);
            float x1_y1_z1 = sqrtf(x1 * x1 + y1 * y1 + z1 * z1);
            float dir_z = z1 + h * x1_y1 / x1_y1_z1;
            float dir_x = (x1_y1 - h * z1 / x1_y1_z1) * x1 / x1_y1 + w * y1 / x1_y1;
            float dir_y = (x1_y1 - h * z1 / x1_y1_z1) * y1 / x1_y1 - w * x1 / x1_y1;
            float norm = sqrtf(dir_x * dir_x + dir_y * dir_y + dir_z * dir_z);
            photo->photoDir.x = dir_x / norm;
            photo->photoDir.y = dir_y / norm;
            photo->photoDir.z = dir_z / norm;
            
            glutWarpPointer(DIM_W/2,DIM_H/2);
            HANDLE_ERROR( cudaMemcpy( ((DataBlock *)dataBlock)->photo, ((DataBlock *)dataBlock)->cpu_photo, sizeof(Photo), cudaMemcpyHostToDevice ) );
        }
    }
}

void exit_func(void *dataBlock){
    //释放用户数据的内存资源
    delete[] ((DataBlock *)dataBlock)->cpu_sphere;
    delete[] ((DataBlock *)dataBlock)->cpu_ground;
    delete[] ((DataBlock *)dataBlock)->cpu_plane;
    delete[] ((DataBlock *)dataBlock)->cpu_photo;
    delete[] ((DataBlock *)dataBlock)->cpu_light;
    delete[] ((DataBlock *)dataBlock)->cpu_image;
    HANDLE_ERROR( cudaFree( ((DataBlock *)dataBlock)->sphere ) );
    HANDLE_ERROR( cudaFree( ((DataBlock *)dataBlock)->ground ) );
    HANDLE_ERROR( cudaFree( ((DataBlock *)dataBlock)->plane ) );
    HANDLE_ERROR( cudaFree( ((DataBlock *)dataBlock)->photo ) );
    HANDLE_ERROR( cudaFree( ((DataBlock *)dataBlock)->light ) );
    HANDLE_ERROR( cudaFree( ((DataBlock *)dataBlock)->image ) );
}

void keyFunc(unsigned char key, int x, int y, void *dataBlock){
    //cout<<"按下了键盘:"<<key<<",键码:";printf("%d\n",key);
    float photoSpeed = 0.2;
    switch (key){
        case 96:{
            if (((DataBlock *)dataBlock)->isLockCursorInWindow == false){
                glutSetCursor(GLUT_CURSOR_NONE);
            }
            else{
                glutSetCursor(GLUT_CURSOR_LEFT_ARROW);
            }
            ((DataBlock *)dataBlock)->isLockCursorInWindow = !(((DataBlock *)dataBlock)->isLockCursorInWindow);
            break;
        }
        case 119:{
            ((DataBlock *)dataBlock)->cpu_photo->center.x += photoSpeed*((DataBlock *)dataBlock)->cpu_photo->photoDir.x;
            ((DataBlock *)dataBlock)->cpu_photo->center.y += photoSpeed*((DataBlock *)dataBlock)->cpu_photo->photoDir.y;
            HANDLE_ERROR( cudaMemcpy( ((DataBlock *)dataBlock)->photo, ((DataBlock *)dataBlock)->cpu_photo, sizeof(Photo), cudaMemcpyHostToDevice ) );
            break;
        }
        case 115:{
            ((DataBlock *)dataBlock)->cpu_photo->center.x -= photoSpeed*((DataBlock *)dataBlock)->cpu_photo->photoDir.x;
            ((DataBlock *)dataBlock)->cpu_photo->center.y -= photoSpeed*((DataBlock *)dataBlock)->cpu_photo->photoDir.y;
            HANDLE_ERROR( cudaMemcpy( ((DataBlock *)dataBlock)->photo, ((DataBlock *)dataBlock)->cpu_photo, sizeof(Photo), cudaMemcpyHostToDevice ) );
            break;
        }
        case 97:{
            ((DataBlock *)dataBlock)->cpu_photo->center.x -= photoSpeed*((DataBlock *)dataBlock)->cpu_photo->photoDir.y;
            ((DataBlock *)dataBlock)->cpu_photo->center.y -= -photoSpeed*((DataBlock *)dataBlock)->cpu_photo->photoDir.x;
            HANDLE_ERROR( cudaMemcpy( ((DataBlock *)dataBlock)->photo, ((DataBlock *)dataBlock)->cpu_photo, sizeof(Photo), cudaMemcpyHostToDevice ) );
            break;
        }
        case 100:{
            ((DataBlock *)dataBlock)->cpu_photo->center.x += photoSpeed*((DataBlock *)dataBlock)->cpu_photo->photoDir.y;
            ((DataBlock *)dataBlock)->cpu_photo->center.y += -photoSpeed*((DataBlock *)dataBlock)->cpu_photo->photoDir.x;
            HANDLE_ERROR( cudaMemcpy( ((DataBlock *)dataBlock)->photo, ((DataBlock *)dataBlock)->cpu_photo, sizeof(Photo), cudaMemcpyHostToDevice ) );
            break;
        }

    }
    //移动相机光源
    //((DataBlock *)dataBlock)->cpu_light->x = ((DataBlock *)dataBlock)->cpu_photo->x;
    //((DataBlock *)dataBlock)->cpu_light->y = ((DataBlock *)dataBlock)->cpu_photo->y;
    //HANDLE_ERROR( cudaMemcpy( ((DataBlock *)dataBlock)->light, ((DataBlock *)dataBlock)->cpu_light, sizeof(Light), cudaMemcpyHostToDevice ) );
}

int main(void){
    DataBlock data;
    //从文本文件中读取数据
    FILE *fid;/*指针*/
    char buf[MAX_LINE];/*数据缓冲区*/
    char *buff, *str;
    float inputData[12];
    int sphereNum = 0, groundNum = 0, planeNum = 0, lightNum = 0, photoNum = 0, imageNum = 0;
    if((fid = fopen("data.txt","r")) == NULL){
        perror("fail to read");exit(1);}
    while(fgets(buf,MAX_LINE,fid) != NULL){
        buff = buf;
        str = strsep(&buff, " ");
        if(!strcmp(str, "Sphere")){sphereNum++;}
        else if(!strcmp(str, "Ground")){groundNum++;}
        else if(!strcmp(str, "Plane")){planeNum++;}
        else if(!strcmp(str, "Light")){lightNum++;}
        else if(!strcmp(str, "Photo")){photoNum++;}
        else if(!strcmp(str, "Image")){imageNum++;}}
    data.cpu_sphere = new Sphere[sphereNum];//坐标x,y,z,半径radius,颜色值r,g,b
    data.cpu_ground = new Ground[groundNum];//坐标z,颜色值r,g,b
    data.cpu_plane = new Plane[planeNum];//坐标,颜色值r,g,b
    data.cpu_light = new Light[lightNum];//坐标x,y,z,颜色值r,g,b
    data.cpu_photo = new Photo[photoNum];//坐标x,y,z
    data.cpu_image = new Image[imageNum];//图片名称
    data.cpu_sphere[0].num = sphereNum;
    data.cpu_ground[0].num = groundNum;
    data.cpu_plane[0].num = planeNum;
    data.cpu_light[0].num = lightNum;
    data.cpu_photo[0].num = photoNum;
    data.cpu_image[0].num = imageNum;
    sphereNum = 0, groundNum = 0, planeNum = 0, lightNum = 0, photoNum = 0, imageNum = 0;
    if((fid = fopen("data.txt","r")) == NULL){
        perror("fail to read");exit(1);}
    while(fgets(buf,MAX_LINE,fid) != NULL){
        buff = buf;
        str = strsep(&buff, " ");
        if(!strcmp(str, "Sphere")){//圆球
            for(int i = 0; i < 7; i++){inputData[i] = atof(strsep(&buff, " "));}
            data.cpu_sphere[sphereNum].setSphere(vec(inputData[0],inputData[1],inputData[2]), 
                inputData[3],vec(inputData[4],inputData[5],inputData[6]));sphereNum++;}
        else if(!strcmp(str, "Ground")){//地面
            for(int i = 0; i < 5; i++){inputData[i] = atof(strsep(&buff, " "));}
            data.cpu_ground[groundNum].setGround(inputData[0], vec(inputData[1],inputData[2],inputData[3]), inputData[4]);groundNum++;}
        else if(!strcmp(str, "Plane")){//平面
            for(int i = 0; i < 11; i++){inputData[i] = atof(strsep(&buff, " "));}
            data.cpu_plane[planeNum].setPlane(vec(inputData[0],inputData[1],inputData[2]), vec(inputData[3],inputData[4],inputData[5]),vec(inputData[6],inputData[7],inputData[8]), inputData[9], inputData[10]);planeNum++;}
        else if(!strcmp(str, "Light")){//光源
            for(int i = 0; i < 6; i++){inputData[i] = atof(strsep(&buff, " "));}
            data.cpu_light[lightNum].setLight(vec(inputData[0],inputData[1],inputData[2]), 
                vec(inputData[3],inputData[4],inputData[5]));lightNum++;}
        else if(!strcmp(str, "Photo")){//相机
            for(int i = 0; i < 10; i++){inputData[i] = atof(strsep(&buff, " "));}
            data.cpu_photo[photoNum].setPhoto(vec(inputData[0],inputData[1],inputData[2]), 
                inputData[3], inputData[4], inputData[5], inputData[6], vec(inputData[7],inputData[8],inputData[9]));photoNum++;}
        else if(!strcmp(str, "Image")){//纹理贴图
            for(int i = 0; i < 1; i++){str = strsep(&buff, "\n");}
            data.cpu_image[imageNum].setImage(str);imageNum++;}
    }
    for(int i = 0; i < data.cpu_sphere->num; i++){
        data.cpu_sphere[i].color.x = rnd( 1.0f )*0.6;
        data.cpu_sphere[i].color.y = rnd( 1.0f )*0.6;
        data.cpu_sphere[i].color.z = rnd( 1.0f )*0.6;}
    data.cpu_sphere[1].color = vec(0,0,0);
    //将数据复制到GPU设备上
    HANDLE_ERROR( cudaMalloc( (void**)&data.sphere, sizeof(Sphere)*sphereNum ) );
    HANDLE_ERROR( cudaMemcpy( data.sphere, data.cpu_sphere, sizeof(Sphere)*sphereNum, cudaMemcpyHostToDevice ) );
    HANDLE_ERROR( cudaMalloc( (void**)&data.ground, sizeof(Ground)*groundNum ) );
    HANDLE_ERROR( cudaMemcpy( data.ground, data.cpu_ground, sizeof(Ground)*groundNum, cudaMemcpyHostToDevice ) );
    HANDLE_ERROR( cudaMalloc( (void**)&data.plane, sizeof(Plane)*planeNum ) );
    HANDLE_ERROR( cudaMemcpy( data.plane, data.cpu_plane, sizeof(Plane)*planeNum, cudaMemcpyHostToDevice ) );
    cout<<lightNum<<endl;
    HANDLE_ERROR( cudaMalloc( (void**)&data.light, sizeof(Light)*lightNum ) );
    HANDLE_ERROR( cudaMemcpy( data.light, data.cpu_light, sizeof(Light)*lightNum, cudaMemcpyHostToDevice ) );
    HANDLE_ERROR( cudaMalloc( (void**)&data.photo, sizeof(Photo) ) );
    HANDLE_ERROR( cudaMemcpy( data.photo, data.cpu_photo, sizeof(Photo), cudaMemcpyHostToDevice ) );
    HANDLE_ERROR( cudaMalloc( (void**)&data.image, sizeof(Image)*imageNum ) );
    HANDLE_ERROR( cudaMemcpy( data.image, data.cpu_image,sizeof(Image)*imageNum, cudaMemcpyHostToDevice ) );
    //定义OpenGL界面功能,光线追踪器
    GPUAnimBitmap  bitmap( DIM_W, DIM_H, &data );
    //bitmap.click_drag( (void (*)(void*,int,int,int,int))point );
    bitmap.passive_motion( (void (*)(void*,int,int))point );
    bitmap.key_func( (void (*)(unsigned char,int,int,void*))keyFunc );
    bitmap.anim_and_exit(
        (void (*)(uchar4*,void*,int))generate_frame, (void (*)(void *))exit_func );
    return 0;
}
