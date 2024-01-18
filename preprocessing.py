def visualize_samples(folder, item):
       with open(f'{folder}/{item}.pkl', 'rb') as file:
         data = pickle.load(file)
       push = data['push']
       
       plt.imshow(data['S_t1'])
       plt.title('S_t')
       plt.plot(push[:,0], push[:,1], 'r')
       plt.scatter(push[1,0], push[1,1], c='k')
       plt.show()
       
       plt.imshow(data['target_t1'])
       plt.title('M_t')
       plt.show()
       
       plt.imshow(data['S_t2'])
       plt.title('S_t+1')
       plt.plot(push[:,0], push[:,1], 'r')
       plt.scatter(push[1,0], push[1,1], c='k')
       plt.show() 
        
       
       plt.imshow(data['target_t2'])
       plt.title('M_t+1')
       plt.show()
   
